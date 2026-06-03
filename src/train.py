from lightning.pytorch.loggers import WandbLogger
from src.models.artsy import ARTSYConfig, ARTSY
import lightning as L
import wandb
from src.constants import ROOT
import os
from src.data.ebm_nlp import SciDocDataConfig, SciDocDatamodule
from dataclasses import dataclass, field, asdict


# TODO group configs
@dataclass
class TrainerConfig:
    precision: str = "16"
    val_check_interval: int = 50
    log_every_n_steps: int = 10
    limit_val_batches: int = 10


@dataclass
class WandbConfig:
    project: str = "pico_retrieval"
    mode: str = "online"


@dataclass
class FullConfig:
    experiment_name: str = "test"
    model: ARTSYConfig = field(default_factory=ARTSYConfig)
    data: SciDocDataConfig = field(default_factory=SciDocDataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def train_artsy(cfg: FullConfig):
    # prepare dirs
    EXPERIMENT_DIR = ROOT / "experiments" / cfg.experiment_name
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # start trainer
    with wandb.init(
        name=cfg.experiment_name,
        config=asdict(cfg),
        dir=EXPERIMENT_DIR,
        **asdict(cfg.wandb),
    ) as run:
        model = ARTSY(cfg.model)
        datamodule = SciDocDatamodule(cfg.data)
        trainer = L.Trainer(
            **asdict(cfg.trainer),
            logger=WandbLogger(save_dir=EXPERIMENT_DIR),
            enable_checkpointing=True,
            default_root_dir=EXPERIMENT_DIR,
        )

        # train
        trainer.fit(model, datamodule)


# def sweep_graphlinker(config):
#     EXPERIMENT_DIR = os.path.join(ROOT, "experiments", config["experiment_name"])
#     os.makedirs(EXPERIMENT_DIR, exist_ok=True)
#
#     def _run():
#         with wandb.init(project=config["project"]) as run:
#             import pprint
#
#             pprint.pprint(run.config)
#             # generate data
#
#             model = GraphLinker(run.config["model"])
#             datamodule = GraphLinkerDataModule(run.config["data"])
#             trainer = L.Trainer(
#                 **run.config["trainer"],
#                 # TODO figure out how wandblogger works with sweeps
#                 logger=WandbLogger(save_dir=EXPERIMENT_DIR),
#                 enable_checkpointing=True,
#                 default_root_dir=EXPERIMENT_DIR,
#             )
#             return trainer.fit(model, datamodule)
#
#     sweep_id = wandb.sweep(
#         sweep=config["sweep_configuration"], project=config["project"]
#     )
#     wandb.agent(sweep_id, function=_run, count=config["run_count"])
