import lightning as L
from torch import nn


class ProbabilisticEncoder(L.LightningModule):
    def __init__(self, in_dim, hidden_dim, shared_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.mu_head = nn.Linear(hidden_dim, shared_dim)
        self.logsigma_head = nn.Linear(hidden_dim, shared_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.mu_head(x), self.logsigma_head(x)
