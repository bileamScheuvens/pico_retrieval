import os

import lightning as L

import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.profilers import (
    PyTorchProfiler,
    SimpleProfiler,
    AdvancedProfiler,
)


from src.constants import ROOT
from src.data.scidocdata import SciDocDatamodule

from lightning.pytorch.loggers import WandbLogger
from src.metrics.trackiologger import TrackioLogger
from src.models.artsy import ARTSY
from src.utils.configs import ExperimentConfig, as_dict


def train_artsy(cfg: ExperimentConfig):
    # prepare dirs
    EXPERIMENT_DIR = ROOT / "experiments" / cfg.experiment_name
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # start trainer
    run = wandb.init(
        name=cfg.experiment_name,
        config=as_dict(cfg),
        project=cfg.logger.project,
        mode=cfg.logger.mode,  # ty:ignore[invalid-argument-type]
    )

    model = ARTSY(cfg.model)
    datamodule = SciDocDatamodule(cfg.data)
    trainer = L.Trainer(
        **as_dict(cfg.trainer),
        logger=WandbLogger(save_dir=EXPERIMENT_DIR, experiment=run),
        default_root_dir=EXPERIMENT_DIR,
        # profiler=AdvancedProfiler(dirpath=EXPERIMENT_DIR, filename="profile"),
        callbacks=[
            ModelCheckpoint(
                dirpath=EXPERIMENT_DIR / "checkpoints", every_n_train_steps=1000
            ),
            EarlyStopping("val/loss"),
            TQDMProgressBar(),
        ],
    )

    # train
    trainer.fit(model, datamodule)
