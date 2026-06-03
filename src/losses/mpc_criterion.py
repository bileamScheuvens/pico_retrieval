import torch
import lightning as L
import torch.nn.functional as F
from src.utils.gaussian_utils import sample_gaussian_tensors


class MpcRetrievalLoss(L.LightningModule):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.log_2pi = torch.log(torch.tensor(2 * torch.pi))

    def forward(
        self,
        query_mean,
        query_logsigma,
        query_z,
        target_mean,
        target_logsigma,
        target_z,
        reduction="mean",
        recall=False,
    ):
        query_mean = F.normalize(query_mean, dim=-1)
        target_mean = F.normalize(target_mean, dim=-1)
        target_samples = sample_gaussian_tensors(target_mean, target_logsigma, 7)

        inv_sigmas = torch.exp(-query_logsigma)
        loc = -0.5 * torch.mean(
            torch.sum(
                (
                    (target_samples.unsqueeze(0) - query_mean.unsqueeze(1).unsqueeze(2))
                    ** 2
                )
                * inv_sigmas.unsqueeze(1).unsqueeze(2),
                dim=-1,
            ),
            dim=-1,
        )
        norm_constant = (-query_mean.shape[-1] / 2) * self.log_2pi - 0.5 * (
            torch.sum(query_logsigma, dim=-1)
        )
        scores = query_z + norm_constant + loc
        scores = scores - torch.max(scores, dim=0, keepdim=True).values

        labels = torch.arange(0, query_mean.shape[0], device=self.device).long()
        loss = F.cross_entropy(scores / self.temperature, labels, reduction=reduction)
        if recall:
            max_scores = torch.max(scores, dim=0).indices - torch.arange(
                0, query_mean.shape[0], device=self.device
            )
            recall = torch.count_nonzero(max_scores == 0) / query_mean.shape[0]
            return loss, recall
        return loss
