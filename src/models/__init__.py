from src.utils.configs import PicoCombinerConfig
from abc import abstractmethod
import lightning as L


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


class PaperEmbedder(L.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    @property
    @abstractmethod
    def embed_type(self) -> str:
        """Must specify whether embedding type is point or probabilistic."""
        pass
