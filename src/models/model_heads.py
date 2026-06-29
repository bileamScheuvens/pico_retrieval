import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


class ProbabilisticEncoder(L.LightningModule):
    def __init__(self, in_dim, hidden_dim, shared_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.mu_head = nn.Linear(hidden_dim, shared_dim)
        self.logsigma_head = nn.Linear(hidden_dim, shared_dim)
        self.clamp_val = 5

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        mu = F.normalize(self.mu_head(x), dim=-1)
        clamped_logsigma = self.logsigma_head(x).clamp(-self.clamp_val, self.clamp_val)
        return torch.stack((mu, clamped_logsigma), dim=-2)  # [B, 2, D]


class PointEncoder(L.LightningModule):
    def __init__(self, in_dim, hidden_dim, shared_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, shared_dim),
        )

    def forward(self, x) -> torch.Tensor:
        return F.normalize(self.model(x), dim=-1)  # [B, D]
