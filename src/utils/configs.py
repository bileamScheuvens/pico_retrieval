from omegaconf import OmegaConf
from enum import Enum
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import lightning as L

cs = ConfigStore()


@dataclass
class SciDocDataConfig:
    batch_size: int
    train_ratio: float
    val_ratio: float
    seed: int
    use_ebm_nlp: bool
    use_pubmed_rct: bool


cs.store(group="data", name="base_data", node=SciDocDataConfig)


@dataclass
class TrainerConfig:
    precision: str | None
    val_check_interval: int | None
    log_every_n_steps: int | None
    limit_val_batches: int | None


cs.store(group="trainer", name="base_trainer", node=TrainerConfig)


@dataclass
class LoggerConfig:
    project: str
    mode: str | None
    space_id: str | None


cs.store(group="logger", name="base_logger", node=LoggerConfig)


class TextEmbedType(Enum):
    PROMPTREPS = "promptreps"
    SENTENCE = "sentence_transformer"


@dataclass
class PubMedPicoConfig:
    base_url: str
    hidden_dim: int
    max_len: int
    shared_dim: int
    text_embed_type: TextEmbedType
    text_embedder_url: str
    use_prob_encoder: bool
    considered_elements: list[str]


cs.store(group="model/pico_extractor", name="base_pubmed_pico", node=PubMedPicoConfig)


@dataclass
class SPECTER2Config:
    adapter: str
    base_url: str
    hidden_dim: int
    max_len: int
    shared_dim: int
    use_prob_encoder: bool


cs.store(group="model/paper_embedder", name="base_specter2", node=SPECTER2Config)


@dataclass
class ARTSYConfig:
    temperature: int
    pico_extractor: PubMedPicoConfig
    paper_embedder: SPECTER2Config
    lr: float
    l2_lambda: float


cs.store(group="model", name="base_model", node=ARTSYConfig)


@dataclass
class ExperimentConfig:
    experiment_name: str
    model: ARTSYConfig
    data: SciDocDataConfig
    trainer: TrainerConfig
    logger: LoggerConfig


# register to hydra
cs.store(name="experiment_config", node=ExperimentConfig)


def as_dict(conf):
    return OmegaConf.to_container(conf, resolve=True)
