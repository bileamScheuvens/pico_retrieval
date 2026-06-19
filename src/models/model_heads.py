import lightning as L
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

    def forward(self, x):
        x = self.backbone(x)
        clamped_logsigma = self.logsigma_head(x).clamp(-self.clamp_val, self.clamp_val)
        return F.normalize(self.mu_head(x), dim=-1), clamped_logsigma


class PointEncoder(L.LightningModule):
    def __init__(self, in_dim, hidden_dim, shared_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, shared_dim),
        )

    def forward(self, x):
        return F.normalize(self.model(x), dim=-1)
