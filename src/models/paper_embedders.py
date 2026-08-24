from typing import TYPE_CHECKING

import lightning as L
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

from src.constants import MODELPATH
from src.models.model_heads import PointEncoder, ProbabilisticEncoder
from src.utils.configs import PaperEmbedderClass, PaperEmbedderConfig

if TYPE_CHECKING:
    from transformers import Tokenizer


def PaperEmbedderFactory(cfg: PaperEmbedderConfig):
    if cfg.model_class == PaperEmbedderClass.SPECTER2:
        return SPECTER2Model(cfg)
    if cfg.model_class == PaperEmbedderClass.SPECTER:
        return SPECTERModel(cfg)


class PaperEmbedder(L.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.tokenizer: Tokenizer = AutoTokenizer.from_pretrained(cfg.base_url)

    def init_head(self, in_dim):
        if self.cfg.use_prob_encoder:
            # turn point embeddings to gaussian
            return ProbabilisticEncoder(
                in_dim,
                self.cfg.hidden_dim,
                self.cfg.shared_dim,
            )
        else:
            return PointEncoder(
                in_dim,
                self.cfg.hidden_dim,
                self.cfg.shared_dim,
            )

    def join_text(self, title, abstract):
        return title + self.tokenizer.sep_token + abstract

    @property
    def embed_type(self):
        if self.cfg.use_prob_encoder:
            return "prob"
        return "point"


class SPECTERModel(PaperEmbedder):
    def __init__(self, cfg: PaperEmbedderConfig):
        super().__init__(cfg)

        # load model
        self.specter = AutoModel.from_pretrained(cfg.base_url)
        self._batch_count = 0
        if cfg.unfreeze_after != 0:
            self.freeze_backbone()
        specter_out_dim = self.specter.config.hidden_size
        self.paper_head = self.init_head(specter_out_dim)

    def freeze_backbone(self):
        for p in self.specter.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.specter.parameters():
            p.requires_grad = True

    def embed_batch(self, batch):
        inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=self.cfg.max_len,
        ).to(self.device)

        output = self.specter(**inputs)

        # take the first token as the embedding
        return output.last_hidden_state[:, 0, :]

    def forward(self, batch):
        if self._batch_count == self.cfg.unfreeze_after:
            self.unfreeze_backbone()
        self._batch_count += 1

        doc_embeddings = self.embed_batch(batch)

        return self.paper_head(doc_embeddings)  # [B, shared_dim]


class SPECTER2Model(PaperEmbedder):
    def __init__(self, cfg: PaperEmbedderConfig):
        super().__init__(cfg)

        # load model as onnx session
        self.specter2_session = ort.InferenceSession(
            MODELPATH / f"specter2_{cfg.adapter}.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        specter_out_dim = self.specter2_session.get_outputs()[0].shape[-1]
        self.paper_head = self.init_head(specter_out_dim)

    def embed_batch(self, batch):
        inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="np",
            return_token_type_ids=False,
            max_length=self.cfg.max_len,
        ).to(self.device)

        output = self.specter2_session.run(["last_hidden_state"], inputs)[0]

        # take the first token as the embedding
        return torch.tensor(output[:, 0, :], device=self.device)  # ty:ignore[invalid-argument-type, not-subscriptable]

    def forward(self, batch):
        doc_embeddings = self.embed_batch(batch)
        return self.paper_head(doc_embeddings)  # [B, shared_dim]
