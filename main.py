from omegaconf import DictConfig, OmegaConf
from src.constants import CONFIGPATH
import hydra
from src.train import train_artsy


# allow interpolation at resolve time
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path=str(CONFIGPATH), config_name="config")
def main(cfg: DictConfig):
    train_artsy(cfg)  # ty:ignore[invalid-argument-type]


if __name__ == "__main__":
    main()
