from omegaconf import DictConfig
from src.constants import CONFIGPATH
import hydra
from src.data.indexing import eval_index_merge


@hydra.main(version_base=None, config_path=str(CONFIGPATH), config_name="eval")
def merge_index(cfg: DictConfig):
    eval_index_merge(cfg.index_name, k_shards=cfg.k_shards)


if __name__ == "__main__":
    merge_index()
