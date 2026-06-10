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
        **kwargs,
    ):
        # TODO normalize in forward
        query = F.normalize(query, dim=-1)
        target = F.normalize(target, dim=-1)
        target_samples = sample_gaussian_tensors(target, target_logsigma, 7)

        inv_sigmas = torch.exp(-query_logsigma)
        loc = -0.5 * torch.mean(
            torch.sum(
                ((target_samples.unsqueeze(0) - query.unsqueeze(1).unsqueeze(2)) ** 2)
                * inv_sigmas.unsqueeze(1).unsqueeze(2),
                dim=-1,
            ),
            dim=-1,
        )
        norm_constant = (-query.shape[-1] / 2) * self.log_2pi - 0.5 * (
            torch.sum(query_logsigma, dim=-1)
        )
        scores = query_z + norm_constant + loc
        scores = scores - torch.max(scores, dim=0, keepdim=True).values

        labels = torch.arange(0, query.shape[0], device=self.device).long()
        loss = F.cross_entropy(scores / self.temperature, labels, reduction=reduction)
        if recall:
            max_scores = torch.max(scores, dim=0).indices - torch.arange(
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
            recall = (preds == labels).mean()
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
        query = F.normalize(query, dim=-1)

        inv_sigmas = torch.exp(-query_logsigma)
        loc = -0.5 * torch.mean(
            torch.sum(
                ((target.unsqueeze(0) - query.unsqueeze(1).unsqueeze(2)) ** 2)
                * inv_sigmas.unsqueeze(1).unsqueeze(2),
                dim=-1,
            ),
            dim=-1,
        )
        norm_constant = (-query.shape[-1] / 2) * self.log_2pi - 0.5 * (
            torch.sum(query_logsigma, dim=-1)
        )
        scores = query_z + norm_constant + loc
        scores = scores - torch.max(scores, dim=0, keepdim=True).values

        labels = torch.arange(0, query.shape[0], device=self.device).long()
        loss = F.cross_entropy(scores / self.temperature, labels, reduction=reduction)
        if recall:
            max_scores = torch.max(scores, dim=0).indices - torch.arange(
                0, query.shape[0], device=self.device
            )
            recall = torch.count_nonzero(max_scores == 0) / query.shape[0]
            return loss, recall
        return loss
