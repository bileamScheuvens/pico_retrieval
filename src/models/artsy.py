from collections import defaultdict
from src.utils.configs import ARTSYConfig
import torch
from src.metrics import mean_l2, mean_sim
from src.metrics.plots import plot_means
from src.models.combiners import get_fitting_combiner
from src.losses import get_fitting_criterion
from src.models.specter2 import SPECTER2Model
from src.models.pubmed_pico import PubMedPicoModel
import lightning as L


class ARTSY(L.LightningModule):
    """Approximate Retrieval Through Semantic Geometry"""

    def __init__(self, cfg=ARTSYConfig):
        super().__init__()
        self.cfg = cfg
        self.pico_embedder = PubMedPicoModel(cfg.pico_extractor)
        self.pico_embed_type = self.pico_embedder.embed_type
        self.paper_embedder = SPECTER2Model(cfg.paper_embedder)
        self.paper_embed_type = self.paper_embedder.embed_type
        self.pico_combiner = get_fitting_combiner(embed_type=self.pico_embed_type)
        # get appropriate criterion depending on if embeddings are probabilistic
        self.retrieval_loss = get_fitting_criterion(
            pico_extractor=self.pico_embedder,
            paper_embedder=self.paper_embedder,
            temperature=1,
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)

    def forward(self, batch):
        # TODO support batching better
        texts = []
        pico_agg = defaultdict(list)
        for title, abstract in zip(*batch):
            text = title + self.paper_embedder.tokenizer.sep_token + abstract  # ty:ignore[unresolved-attribute]
            texts.append(text)
            pico_individual_embeddings = self.pico_embedder(
                text
            )  # [num_pico_elements, shared_dim]
            self.pico_combiner(pico_individual_embeddings, agg=pico_agg)

        preds = {
            "pico_means": torch.stack(pico_agg["mean"]),
        }

        if self.paper_embed_type == "prob":
            paper_means, paper_variances = self.paper_embedder(texts)
            preds["paper_means"] = paper_means
            preds["paper_variances"] = paper_variances
        else:
            preds["paper_means"] = self.paper_embedder(texts)

        if self.pico_embed_type == "prob":
            preds["pico_variances"] = torch.stack(pico_agg["variance"])
            preds["pico_zs"] = torch.stack(pico_agg["log_z"])

        return preds

    def _compute_loss(
        self,
        paper_means,
        pico_means,
        paper_variances=None,
        pico_variances=None,
        pico_zs=None,
        prefix="train",
    ):
        # TODO bidirectional retrieval loss
        retrieval_loss, recall = self.retrieval_loss.forward(
            query=pico_means,
            query_logsigma=pico_variances,
            query_z=pico_zs,
            target=paper_means,
            target_logsigma=paper_variances,
            target_z=None,
            recall=True,
        )
        loss = retrieval_loss
        metrics = {
            f"{prefix}/retrieval_loss": retrieval_loss,
            f"{prefix}/recall": recall,
        }
        if pico_variances is not None:
            l2_loss = self.cfg.l2_lambda * (pico_variances**2).sum()
            metrics[f"{prefix}/l2_loss"] = l2_loss
            loss += l2_loss
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
