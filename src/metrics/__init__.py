import torch
import torch.nn.functional as F


def mean_l2(paper_means, pico_means):
    """Compute mean l2 distance for correct and incorrect matches"""
    diff = torch.cdist(paper_means, pico_means, p=2)
    offdiag = diff[~torch.eye(paper_means.shape[0], dtype=torch.bool)]
    return diff.diagonal().mean(), offdiag.mean()


def mean_sim(paper_means, pico_means):
    """Compute mean cosine similarity for correct and incorrect matches"""
    sim = F.cosine_similarity(paper_means.unsqueeze(0), pico_means.unsqueeze(1), dim=-1)
    offdiag = sim[~torch.eye(paper_means.shape[0], dtype=torch.bool)]
    return sim.diagonal().mean(), offdiag.mean()
