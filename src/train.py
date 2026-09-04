import os

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

import wandb
from src.constants import ROOT
from src.data.scidocdata import SciDocDatamodule
from src.models.artsy import ARTSY
from src.utils.configs import ExperimentConfig, as_dict, get_wandb_names


def train_artsy(cfg: ExperimentConfig):
    # prepare dirs
    EXPERIMENT_DIR = ROOT / "experiments" / cfg.experiment_name
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # start trainer
    experiment_name, group_name = get_wandb_names(cfg)
    run = wandb.init(
        name=experiment_name,
        group=group_name,
        config=as_dict(cfg),
        project=cfg.logger.project,
        mode=cfg.logger.mode,  # ty:ignore[invalid-argument-type]
    )

    model = ARTSY(cfg.model)
    datamodule = SciDocDatamodule(cfg.data)
    # early stopping only once after 1000 batches, otherwise 2 epochs (more if overfitting on purpose)
    patience = (
        int(1000 / cfg.trainer.val_check_interval)
        if cfg.trainer.val_check_interval
        else 2 + cfg.trainer.overfit_batches * 1000
    )
    trainer = L.Trainer(
        **as_dict(cfg.trainer),
        logger=WandbLogger(save_dir=EXPERIMENT_DIR, experiment=run),
        default_root_dir=EXPERIMENT_DIR,
        callbacks=[
            ModelCheckpoint(
                dirpath=EXPERIMENT_DIR / "checkpoints",
                every_n_train_steps=250,
                save_on_exception=True,
            ),
            EarlyStopping("val/loss", patience=patience),
            TQDMProgressBar(),
        ],
    )

    # train
    trainer.fit(model, datamodule, ckpt_path=cfg.model.ckpt_path, weights_only=False)
    # cleanup (necessary for multirun)
    wandb.finish()
    del model
    del trainer
    del datamodule
