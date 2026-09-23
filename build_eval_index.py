from src.models.artsy import ARTSY
from omegaconf import DictConfig
from src.constants import CONFIGPATH
import hydra
from src.data.indexing import eval_index_shard


@hydra.main(version_base=None, config_path=str(CONFIGPATH), config_name="eval")
def index_shard(cfg: DictConfig):
    model = ARTSY.load_from_checkpoint(
        cfg.model.ckpt_path, weights_only=False, strict=False
    )
    model.eval()
    eval_index_shard(model, cfg.index_name, shard_i=cfg.shard_i, k_shards=cfg.k_shards)


if __name__ == "__main__":
    index_shard()
