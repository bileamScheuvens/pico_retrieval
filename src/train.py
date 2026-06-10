from src.utils.configs import ExperimentConfig, as_dict
from lightning.pytorch.loggers import WandbLogger
from src.models.artsy import ARTSY
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from src.constants import ROOT
import os
from src.data.scidocdata import SciDocDatamodule


def train_artsy(cfg: ExperimentConfig):
    # prepare dirs
    EXPERIMENT_DIR = ROOT / "experiments" / cfg.experiment_name
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # start trainer
    with wandb.init(
        name=cfg.experiment_name,
        config=as_dict(cfg),
        dir=EXPERIMENT_DIR,
        **as_dict(cfg.wandb),
    ) as run:
        model = ARTSY(cfg.model)
        datamodule = SciDocDatamodule(cfg.data)
        trainer = L.Trainer(
            **as_dict(cfg.trainer),
            logger=WandbLogger(save_dir=EXPERIMENT_DIR),
            enable_checkpointing=True,
            default_root_dir=EXPERIMENT_DIR,
            callbacks=[
                ModelCheckpoint(
                    dirpath=EXPERIMENT_DIR / "checkpoints", every_n_train_steps=1000
                ),
                EarlyStopping("val/loss"),
            ],
        )

        # train
        trainer.fit(model, datamodule)
