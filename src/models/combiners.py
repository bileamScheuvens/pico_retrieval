from typing import Callable

import lightning as L
import torch
from torch.distributions import MultivariateNormal
from torch.nn.functional import normalize

from src.utils.configs import PicoAggType, PicoCombinerConfig


def AggFactory(agg_type: PicoAggType, cfg: PicoCombinerConfig) -> Callable:
    if agg_type == PicoAggType.SUM:
        return lambda x: normalize(torch.sum(x, dim=0), dim=0)
    if agg_type == PicoAggType.MEAN:
        return lambda x: normalize(torch.mean(x, dim=0), dim=0)
    if agg_type == PicoAggType.HADAMARD:
        return lambda x: normalize(torch.prod(x, dim=0), dim=0)
    if agg_type == PicoAggType.ATTN:
        return AttentionCombiner(cfg)
    if agg_type == PicoAggType.GAUSSIAN:
        return MpcCombiner(cfg)
    if agg_type == PicoAggType.MLP:
        raise NotImplementedError


class Combiner(L.LightningModule):
    """Combine individual pico embeddings. First within category (ideally logical OR), then between categories (logical AND)."""

    def __init__(self, cfg: PicoCombinerConfig):
        super().__init__()
        self.cfg = cfg
        self.intra_combiner = AggFactory(cfg.intra_agg, cfg)
        self.inter_combiner = AggFactory(cfg.inter_agg, cfg)

    def forward(self, embeds, labels):
        if self.cfg.use_prob_encoder:
            return self.forward_prob(embeds, labels)
        else:
            return self.forward_point(embeds, labels)

    def forward_point(self, embeds, labels):
        # embeds shape: [n_elements, shared_dim]
        inter_embeds = []
        for label in set(labels):
            current_embeds = [e for (e, e_l) in zip(embeds, labels) if e_l == label]
            inter_embeds.append(self.inter_combiner(torch.stack(current_embeds)))
        return {"mean": self.intra_combiner(torch.stack(inter_embeds))}  # [shared_dim]

    def forward_prob(self, embeds, labels):
        # embeds shape: [n_elements, 2, shared_dim]
        intra_embeds = []
        log_z = torch.zeros(1).to(self.device)

        for label in set(labels):
            current_embeds = [e for (e, e_l) in zip(embeds, labels) if e_l == label]
            intra_embed = self.intra_combiner(current_embeds)
            intra_embeds.append(
                torch.stack((intra_embed["mean"], intra_embed["variance"]))
            )
            log_z += intra_embed["log_z"]
        # stack all
        return self.inter_combiner(intra_embeds, log_z=log_z)


class HadamardCombiner(L.LightningModule):
    def __init__(self, cfg: PicoCombinerConfig):
        self.cfg = cfg


class AttentionCombiner(L.LightningModule):
    def __init__(self, cfg: PicoCombinerConfig):
        pass


class MpcCombiner(L.LightningModule):
    """Multimodal Probabilistic Composer from neculai2022probabilistic"""

    def __init__(self, cfg: PicoCombinerConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, embeddings, log_z=torch.zeros(1)):
        if len(embeddings) == 1:
            combined = {
                "mean": embeddings[0][0],
                "variance": embeddings[0][1],
                "log_z": log_z.to(self.device),
            }
        else:
            prior_mean = embeddings[0][0]
            prior_variance = embeddings[0][1]
            log_z_total = log_z.to(self.device)
            for i in range(1, len(embeddings)):
                posterior_mean, posterior_variance, log_z = product_2_gaussians(
                    prior_mean,
                    prior_variance,
                    embeddings[i][0],
                    embeddings[i][1],
                )
                log_z_total += log_z
                prior_mean = posterior_mean
                prior_variance = posterior_variance

            combined = {
                "mean": posterior_mean,
                "variance": posterior_variance.squeeze(-2),
                "log_z": log_z_total,
            }
        return combined


def product_2_gaussians(mean1, variance1, mean2, variance2):
    if len(mean1.shape) == 1:
        mean1 = mean1.unsqueeze(0)
    if len(variance1.shape) == 1:
        variance1 = variance1.unsqueeze(0)
    if len(mean2.shape) == 1:
        mean2 = mean2.unsqueeze(0)
    if len(variance2.shape) == 1:
        variance2 = variance2.unsqueeze(0)
    variance1 = torch.exp(variance1)
    variance2 = torch.exp(variance2)

    target_mean = mean2
    target_variance = variance2

    inv_variance1 = 1 / variance1
    inv_target_variance = 1 / target_variance
    C = torch.diag_embed(1 / (inv_variance1 + inv_target_variance))
    c = torch.matmul(
        C,
        torch.matmul(torch.diag_embed(inv_variance1), mean1[:, :, None])
        + torch.matmul(torch.diag_embed(inv_target_variance), target_mean[:, :, None]),
    ).squeeze()
    log_z = MultivariateNormal(
        target_mean, torch.diag_embed(variance1 + target_variance + 1e-6)
    ).log_prob(mean1)
    C = torch.diagonal(C, dim1=-2, dim2=-1)
    C = torch.log(C)

    return c, C, log_z
