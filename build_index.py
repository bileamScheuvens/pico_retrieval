from src.models.artsy import ARTSY
from omegaconf import DictConfig
from src.constants import CONFIGPATH
import hydra
from src.data.indexing import PICOIndex


@hydra.main(version_base=None, config_path=str(CONFIGPATH), config_name="config")
def prepare_index(cfg: DictConfig):
    model = ARTSY(cfg.model)
    index = PICOIndex(model.pico_extractor)
    first_doc = list(index.pmid2idx["Patient"].keys())[0]
    index.get_similar_doc(first_doc)


if __name__ == "__main__":
    prepare_index()
