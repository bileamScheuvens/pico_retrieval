import torch
from src.metrics.plots import plot_means
from src.models.mpc import MpcCombiner
from dataclasses import dataclass, field
from src.losses.mpc_criterion import MpcRetrievalLoss
from src.models.specter2 import SPECTER2Model, SPECTER2Config
from src.models.pubmed_pico import PubMedPicoModel, PubMedPicoConfig
import lightning as L


@dataclass
class ARTSYConfig:
    pico_extractor: PubMedPicoConfig = field(default_factory=PubMedPicoConfig)
    paper_embedder: SPECTER2Config = field(default_factory=SPECTER2Config)
    lr: float = 5e-3
    l2_lambda: float = 1e-4


class ARTSY(L.LightningModule):
    """Approximate Retrieval Through Semantic Geometry"""

    def __init__(self, cfg=ARTSYConfig):
        super().__init__()
        self.cfg = cfg
        self.pico_embedder = PubMedPicoModel(cfg.pico_extractor)
        self.paper_embedder = SPECTER2Model(cfg.paper_embedder)
        self.mpc_combiner = MpcCombiner()
        self.retrieval_loss = MpcRetrievalLoss(temperature=1)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)

    def forward(self, batch):
        # TODO support batching better
        texts = []
        pico_means, pico_variances, pico_zs = [], [], []
        for title, abstract in zip(*batch):
            text = title + self.paper_embedder.tokenizer.sep_token + abstract
            texts.append(text)
            pico_individual_embeddings = self.pico_embedder(
                text
            )  # [num_pico_elements, shared_dim]
            pico_mean, pico_variance, pico_z = self.mpc_combiner(
                pico_individual_embeddings
            )
            pico_means.append(pico_mean)
            pico_variances.append(pico_variance)
            pico_zs.append(pico_z)
        paper_means, paper_variances = self.paper_embedder(texts)
        return (
            paper_means,
            paper_variances,
            torch.stack(pico_means),
            torch.stack(pico_variances),
            torch.stack(pico_zs),
        )

    def _compute_loss(
        self,
        paper_means,
        paper_variances,
        pico_means,
        pico_variances,
        pico_zs,
        prefix="train",
    ):
        # TODO bidirectional retrieval loss
        retrieval_loss, recall = self.retrieval_loss.forward(
            pico_means,
            pico_variances,
            pico_zs,
            paper_means,
            paper_variances,
            None,
            recall=True,
        )
        l2_loss = self.cfg.l2_lambda * (pico_variances**2).sum()
        loss = retrieval_loss + l2_loss
        metrics = {
            f"{prefix}/loss": loss,
            f"{prefix}/retrieval_loss": retrieval_loss,
            f"{prefix}/l2_loss": l2_loss,
            f"{prefix}/recall": recall,
        }
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self._compute_loss(*self(batch))
        self.log_dict(metrics, batch_size=len(batch[0]))
        return metrics["train/loss"]

    def on_validation_start(self):
        self.umap_buffer = {"pico_means": [], "paper_means": []}
        self.val_batch_count = 0

    def validation_step(self, batch, batch_idx):
        self.val_batch_count += 1
        paper_means, paper_variances, pico_means, pico_variances, pico_zs = self(batch)
        metrics = self._compute_loss(
            paper_means,
            paper_variances,
            pico_means,
            pico_variances,
            pico_zs,
            prefix="val",
        )
        self.log_dict(metrics, batch_size=len(batch[0]))
        if self.val_batch_count < 5:
            self.umap_buffer["pico_means"].append(pico_means)
            self.umap_buffer["paper_means"].append(paper_means)
        return metrics["val/loss"]

    def on_validation_end(self):
        self.logger.experiment.log(  # ty:ignore[unresolved-attribute]
            {
                "mean_umap": plot_means(
                    torch.cat(self.umap_buffer["pico_means"]),
                    torch.cat(self.umap_buffer["paper_means"]),
                )
            }
        )
