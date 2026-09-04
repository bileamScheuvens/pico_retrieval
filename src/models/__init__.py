from abc import abstractmethod

import lightning as L

from src.models.text_embedders import PromptRepsModel, SentenceTransformerModel
from src.utils.configs import TextEmbedType


class PicoExtractor(L.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg


class PicoProjector(L.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    @property
    @abstractmethod
    def embed_type(self) -> str:
        """Must specify whether embedding type is point or probabilistic."""
        pass


def TextEncoderFactory(cfg):
    if cfg.text_embed_type == TextEmbedType.SENTENCE:
        return SentenceTransformerModel(cfg.text_embedder.value)
    if cfg.text_embed_type == TextEmbedType.PROMPTREPS:
        return PromptRepsModel(cfg.text_embedder.value)
    raise NotImplementedError(f"No text encoder of type {cfg.text_embed_type}")
