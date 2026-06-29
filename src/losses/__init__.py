from .criteria import MpcRetrievalLoss, LikelihoodRetrievalLoss, InfoNCERetrievalLoss


def get_fitting_criterion(
    pico_embed_type: str, paper_embed_type: str, temperature: float
):
    if pico_embed_type == "prob" and paper_embed_type == "prob":
        return MpcRetrievalLoss(temperature)
    if pico_embed_type == "prob" and paper_embed_type == "point":
        return LikelihoodRetrievalLoss(temperature)
    else:
        return InfoNCERetrievalLoss(temperature)
