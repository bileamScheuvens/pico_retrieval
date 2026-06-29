from src.models.pubmed_pico import PubMedPicoModel
from src.data.indexing import PICOIndex

cfg = None
index = PICOIndex(model=PubMedPicoModel(cfg))

