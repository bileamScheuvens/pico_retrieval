import lightning as L
import torch
import torch.nn.functional as F

from src.utils.gaussian_utils import sample_gaussian_tensors


class MpcRetrievalLoss(L.LightningModule):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.log_2pi = torch.log(torch.tensor(2 * torch.pi))

    def forward(
        self,
        query,
        query_logsigma,
        query_z,
        target,
        target_logsigma,
        target_z,
        reduction="mean",
        recall=False,
        n_samples=10,
        **kwargs,
    ):

        # TODO
        target_samples = sample_gaussian_tensors(
            target, target_logsigma, n_samples
        )  # [B, n_samples, shared_dim]

        inv_sigmas = torch.exp(-query_logsigma)

        norm_constant = (-query.shape[-1] / 2) * self.log_2pi - 0.5 * (
            torch.sum(query_logsigma, dim=-1, keepdim=True)
        )

        diff = torch.mean(
            (target_samples.unsqueeze(0) - query.unsqueeze(1).unsqueeze(2)) ** 2, dim=-2
        )
        log_prob = -0.5 * (diff * inv_sigmas).sum(-1) + norm_constant
        scores = query_z + log_prob
        # scores = scores - torch.max(scores, dim=0, keepdim=True).values

        labels = torch.arange(0, query.shape[0], device=self.device).long()
        loss = F.cross_entropy(scores / self.temperature, labels, reduction=reduction)
        if recall:
            max_scores = torch.max(scores, dim=1).indices - torch.arange(
                0, query.shape[0], device=self.device
            )

            recall = torch.count_nonzero(max_scores == 0) / query.shape[0]
            return loss, recall
        return loss


class InfoNCERetrievalLoss(L.LightningModule):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query,
        target,
        reduction="mean",
        recall=False,
        **kwargs,
    ):

        scores = query @ target.T

        labels = torch.arange(query.shape[0], device=self.device).long()
        loss = F.cross_entropy(scores / self.temperature, labels, reduction=reduction)
        if recall:
            preds = scores.argmax(dim=1)
            recall = (preds == labels).float().mean()
            return loss, recall
        return loss


class LikelihoodRetrievalLoss(L.LightningModule):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.log_2pi = torch.log(torch.tensor(2 * torch.pi))

    def forward(
        self,
        query,
        query_logsigma,
        query_z,
        target,
        reduction="mean",
        recall=False,
        **kwargs,
    ):
        inv_sigmas = torch.exp(-query_logsigma)

        norm_constant = (-query.shape[-1] / 2) * self.log_2pi - 0.5 * (
            torch.sum(query_logsigma, dim=-1, keepdim=True)
        )

        diff = (target.unsqueeze(0) - query.unsqueeze(1)) ** 2
        log_prob = -0.5 * (diff * inv_sigmas).sum(-1) + norm_constant
        scores = query_z + log_prob
        # scores = scores - torch.max(scores, dim=0, keepdim=True).values

        labels = torch.arange(0, query.shape[0], device=self.device).long()
        loss = F.cross_entropy(scores / self.temperature, labels, reduction=reduction)
        if recall:
            max_scores = torch.max(scores, dim=1).indices - torch.arange(
                0, query.shape[0], device=self.device
            )

            recall = torch.count_nonzero(max_scores == 0) / query.shape[0]
            return loss, recall

        return loss
