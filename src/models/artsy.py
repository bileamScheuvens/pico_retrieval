from src.models.combiners import Combiner
from collections import defaultdict

import lightning as L
import torch

from src.losses import get_fitting_criterion
from src.metrics import mean_l2, mean_sim
from src.metrics.plots import plot_means
from src.models.pubmed_pico import PubMedPicoModel, PubMedPicoProjector
from src.models.specter2 import SPECTER2Model
from src.utils.configs import ARTSYConfig


class Dummy(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(5, 5)

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        return {"loss": torch.ones(1, requires_grad=True)}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


class ARTSY(L.LightningModule):
    """Approximate Retrieval Through Semantic Geometry"""

    def __init__(self, cfg=ARTSYConfig):
        super().__init__()
        self.cfg = cfg
        self.strict_loading = False

        # PICO branch components
        self.pico_extractor = PubMedPicoModel(cfg.pico_extractor)
        self.extract_pico = self.pico_extractor.extract_pico
        self.pico_projector = PubMedPicoProjector(
            cfg.pico_extractor, self.pico_extractor.embed_dim
        )
        self.pico_embed_type = self.pico_projector.embed_type

        # Paper branch components
        self.paper_embedder = SPECTER2Model(cfg.paper_embedder)
        self.paper_embed_type = self.paper_embedder.embed_type
        self.pico_combiner = Combiner(cfg.combiner)
        # get appropriate criterion depending on if embeddings are probabilistic
        self.retrieval_loss = get_fitting_criterion(
            pico_embed_type=self.pico_embed_type,
            paper_embed_type=self.paper_embed_type,
            temperature=1,
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)

    def join_text(self, title, abstract):
        return self.paper_embedder.join_text(title, abstract)

    @torch.no_grad
    def embed_query(self, pico):
        pico_individual, labels = self.pico_extractor.encode_pico(pico)
        pico_projected = self.pico_projector(pico_individual)
        means = self.pico_combiner(pico_projected, labels)["mean"]
        return means.unsqueeze(0).cpu()

    @torch.no_grad()
    def embed_paper(self, title, abstract):
        paper_embed = self.paper_embedder(self.join_text(title, abstract))
        return paper_embed.cpu()

    def forward(self, batch):
        # TODO support batching better
        texts = []
        # aggregates combiner outputs (just mean or mean, var, log_z)
        pico_agg = defaultdict(list)
        for title, abstract in zip(*batch):
            text = self.join_text(title, abstract)
            texts.append(text)
            pico_individual, pico_labels = self.pico_extractor(
                text
            )  # [n_pico_elements, shared_dim]
            pico_projected = self.pico_projector(pico_individual)

            combined = self.pico_combiner(pico_projected, pico_labels)
            for k, v in combined.items():
                pico_agg[k].append(v)

        preds = {
            "pico_means": torch.stack(pico_agg["mean"]),
        }

        if self.paper_embed_type == "prob":
            paper_embedding = self.paper_embedder(texts)  # [B, 2, shared_dim]
            preds["paper_means"] = paper_embedding[..., 0, :]
            preds["paper_logvars"] = paper_embedding[..., 1, :]

        else:
            preds["paper_means"] = self.paper_embedder(texts)

        if self.pico_embed_type == "prob":
            preds["pico_logvars"] = torch.stack(pico_agg["variance"])
            preds["pico_zs"] = torch.stack(pico_agg["log_z"])

        return preds

    def _compute_loss(
        self,
        paper_means,
        pico_means,
        paper_logvars=None,
        pico_logvars=None,
        pico_zs=None,
        prefix="train",
    ):
        # TODO bidirectional retrieval loss
        retrieval_loss, recall = self.retrieval_loss.forward(
            query=pico_means,
            query_logsigma=pico_logvars,
            query_z=pico_zs,
            target=paper_means,
            target_logsigma=paper_logvars,
            target_z=None,
            recall=True,
        )
        loss = retrieval_loss
        metrics = {
            f"{prefix}/retrieval_loss": retrieval_loss,
            f"{prefix}/recall": recall,
        }
        if pico_logvars is not None:
            l2_pico = self.cfg.l2_lambda * ((pico_logvars) ** 2).mean()
            metrics[f"{prefix}/l2_pico"] = l2_pico
            metrics[f"{prefix}/mean_pico_var"] = pico_logvars.exp().mean()
            loss += l2_pico
        if paper_logvars is not None:
            l2_paper = self.cfg.l2_lambda * ((paper_logvars) ** 2).mean()
            metrics[f"{prefix}/l2_paper"] = l2_paper
            metrics[f"{prefix}/mean_paper_var"] = paper_logvars.exp().mean()
            loss += l2_paper
        metrics[f"{prefix}/loss"] = loss
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self._compute_loss(**self(batch))
        self.log_dict(metrics, batch_size=len(batch[0]))
        return metrics["train/loss"]

    def on_validation_start(self):
        self.umap_buffer = {"pico_means": [], "paper_means": [], "titles": []}
        self.val_batch_count = 0

    def validation_step(self, batch, batch_idx):
        self.val_batch_count += 1
        preds = self(batch)
        metrics = self._compute_loss(
            **preds,
            prefix="val",
        )
        mean_l2_correct, mean_l2_incorrect = mean_l2(
            preds["paper_means"], preds["pico_means"]
        )
        mean_sim_correct, mean_sim_incorrect = mean_sim(
            preds["paper_means"], preds["pico_means"]
        )
        metrics["val/mean_l2_correct"] = mean_l2_correct
        metrics["val/mean_l2_incorrect"] = mean_l2_incorrect
        metrics["val/mean_sim_correct"] = mean_sim_correct
        metrics["val/mean_sim_incorrect"] = mean_sim_incorrect
        self.log_dict(metrics, batch_size=len(batch[0]))
        if self.val_batch_count < 3:
            self.umap_buffer["pico_means"].append(preds["pico_means"])
            self.umap_buffer["paper_means"].append(preds["paper_means"])
            self.umap_buffer["titles"].extend(batch[0])
        return metrics["val/loss"]

    def on_validation_end(self):
        fig = plot_means(
            torch.cat(self.umap_buffer["pico_means"]),
            torch.cat(self.umap_buffer["paper_means"]),
            paper_titles=self.umap_buffer["titles"],
        )
        # fig_img = Image.open(io.BytesIO(fig.to_image(format="png")))
        self.logger.experiment.log({"mean_umap": fig})  # ty:ignore[unresolved-attribute]
