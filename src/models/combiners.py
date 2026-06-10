import torch
from torch.distributions import MultivariateNormal
import lightning as L


def get_fitting_combiner(embed_type):
    if embed_type == "prob":
        return MpcCombiner()
    return LinearCombiner()


class LinearCombiner(L.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings):
        raise NotImplementedError


class MpcCombiner(L.LightningModule):
    """Multimodal Probabilistic Composer from neculai2022probabilistic"""

    def __init__(self):
        super().__init__()

    def forward(self, embeddings, agg=None):
        """Combine embeddings. Optionally takes aggregator instead of returning result."""
        if len(embeddings) == 1:
            combined = {
                "mean": embeddings[0][0],
                "variance": embeddings[0][1],
                "log_z": torch.zeros(1).to(embeddings[0][0].device),
            }
        else:
            prior_mean = embeddings[0][0]
            prior_variance = embeddings[0][1]
            log_z_total = 0
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
        if agg is not None:
            for k, v in combined.items():
                agg[k].append(v)
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
