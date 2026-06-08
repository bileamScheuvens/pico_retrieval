from src.models import PaperEmbedder, PicoExtractor
from .criteria import MpcRetrievalLoss, LikelihoodRetrievalLoss, InfoNCERetrievalLoss


def get_fitting_criterion(
    pico_extractor: PicoExtractor, paper_embedder: PaperEmbedder, temperature: float
):
    if pico_extractor.embed_type == "prob" and paper_embedder.embed_type == "prob":
        return MpcRetrievalLoss(temperature)
    if pico_extractor.embed_type == "prob" and paper_embedder.embed_type == "point":
        return LikelihoodRetrievalLoss(temperature)
    else:
        return InfoNCERetrievalLoss(temperature)
