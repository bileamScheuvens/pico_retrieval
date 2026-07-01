from pathlib import Path
from typing import Optional, Union
from omegaconf import OmegaConf
from enum import Enum
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

cs = ConfigStore()


class SamplerType(Enum):
    RANDOM = "random_sampler"
    HARDNEGATIVE = "hard_negative_sampler"


@dataclass
class SamplerConfig:
    sampler_type: SamplerType


@dataclass
class HardNegativeConfig(SamplerConfig):
    sampler_type: SamplerType
    corpus: str
    extractor_model: str
    n_candidates: int
    n_subcandidates: int
    n_anchors: int
    n_per_anchor: int


cs.store(group="data/sampler", name="base_sampler", node=SamplerConfig)
cs.store(group="data/sampler", name="base_hardnegative", node=HardNegativeConfig)


@dataclass
class SciDocDataConfig:
    batch_size: int
    train_ratio: float
    val_ratio: float
    seed: int
    use_ebm_nlp: bool
    use_pubmed_rct: bool
    sampler: SamplerConfig


cs.store(group="data", name="base_data", node=SciDocDataConfig)


@dataclass
class TrainerConfig:
    precision: Optional[str] = None
    val_check_interval: Optional[int] = None
    log_every_n_steps: Optional[int] = None
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    overfit_batches: int = 0


cs.store(group="trainer", name="base_trainer", node=TrainerConfig)


@dataclass
class LoggerConfig:
    project: str
    mode: Optional[str] = None
    space_id: Optional[str] = None


cs.store(group="logger", name="base_logger", node=LoggerConfig)


class TextEmbedType(Enum):
    PROMPTREPS = "promptreps"
    SENTENCE = "sentence_transformer"


class TextEmbedder(Enum):
    PUBMEDBERT = "neuml/pubmedbert-base-embeddings"
    QWEN = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class PubMedPicoConfig:
    base_url: str
    hidden_dim: int
    max_len: int
    shared_dim: int
    text_embed_type: TextEmbedType
    text_embedder: TextEmbedder
    use_prob_encoder: bool
    considered_elements: list[str]
    cache_extraction: bool = False
    pico_dropout: float = 0.0


cs.store(group="model/pico_extractor", name="base_pubmed_pico", node=PubMedPicoConfig)


# TODO rename these more abstractly
@dataclass
class PaperEmbedderConfig:
    adapter: str
    base_url: str
    text_embed_type: Optional[TextEmbedType]
    text_embedder: Optional[TextEmbedder]
    hidden_dim: int
    max_len: int
    shared_dim: int
    use_prob_encoder: bool


cs.store(group="model/paper_embedder", name="base_specter2", node=PaperEmbedderConfig)


class PicoAggType(Enum):
    MEAN = "mean"
    SUM = "sum"
    ATTN = "attention"
    GAUSSIAN = "gaussian"
    HADAMARD = "hadamard"
    MLP = "mlp"


@dataclass
class PicoCombinerConfig:
    intra_agg: PicoAggType
    inter_agg: PicoAggType
    n_heads: Optional[int] = 8
    use_prob_encoder: bool = False


cs.store(group="model/combiner", name="base_combiner", node=PicoCombinerConfig)


@dataclass
class ARTSYConfig:
    temperature: int
    pico_extractor: PubMedPicoConfig
    paper_embedder: PaperEmbedderConfig
    combiner: PicoCombinerConfig
    lr: float
    l2_lambda: float
    ckpt_path: Optional[Union[str, Path]] = None


cs.store(group="model", name="base_model", node=ARTSYConfig)


@dataclass
class EvalConfig:
    model: ARTSYConfig
    data: SciDocDataConfig
    index_name: str = "index"


cs.store(name="base_eval", node=EvalConfig)


@dataclass
class ExperimentConfig:
    experiment_name: str
    model: ARTSYConfig
    data: SciDocDataConfig
    trainer: TrainerConfig
    logger: LoggerConfig


cs.store(name="experiment_config", node=ExperimentConfig)


def as_dict(conf):
    return OmegaConf.to_container(conf, resolve=True)
