from src.models.artsy import ARTSY
from omegaconf import DictConfig
from src.constants import CONFIGPATH
import hydra
from src.data.indexing import eval_index_merge, eval_index_shard


@hydra.main(version_base=None, config_path=str(CONFIGPATH), config_name="eval")
def index_shard(cfg: DictConfig):
    model = ARTSY(cfg.model)
    eval_index_shard(model, cfg.index_name, shard_i=cfg.shard_i, k_shards=cfg.k_shards)


if __name__ == "__main__":
    index_shard()
