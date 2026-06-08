from abc import abstractmethod
import lightning as L


class PicoExtractor(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def embed_type(self) -> str:
        """Must specify whether embedding type is point or probabilistic."""
        pass


class PaperEmbedder(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def embed_type(self) -> str:
        """Must specify whether embedding type is point or probabilistic."""
        pass
