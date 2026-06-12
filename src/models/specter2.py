from src.constants import MODELPATH
from torch import nn
import torch

import onnxruntime as ort
from transformers import AutoTokenizer

from src.models import PaperEmbedder
from src.models.prob_encoder import ProbabilisticEncoder
from src.utils.configs import SPECTER2Config


class SPECTER2Model(PaperEmbedder):
    def __init__(self, cfg: SPECTER2Config):
        super().__init__()
        self.cfg = cfg
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_url)

        # load base model and freeze
        self.specter2_session = ort.InferenceSession(
            MODELPATH / f"specter2_{cfg.adapter}.onnx"
        )
        self.specter_out_dim = self.specter2_session.get_outputs()[0].shape[-1]

        if cfg.use_prob_encoder:
            # turn point embeddings to gaussian
            self.paper_head = ProbabilisticEncoder(
                self.specter_out_dim,
                cfg.hidden_dim,
                cfg.shared_dim,
            )
        else:
            self.paper_head = nn.Sequential(
                nn.Linear(self.specter_out_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, cfg.shared_dim),
            )

    @property
    def embed_type(self):
        if self.cfg.use_prob_encoder:
            return "prob"
        return "point"

    def embed_paper(self, paper):
        inputs = self.tokenizer(
            paper,
            padding=True,
            truncation=True,
            return_tensors="np",
            return_token_type_ids=False,
            max_length=self.cfg.max_len,
        ).to(self.device)  # ty:ignore[call-non-callable]

        output = self.specter2_session.run(["last_hidden_state"], inputs)[0]

        # take the first token in the batch as the embedding
        return torch.tensor(output[:, 0, :])  # ty:ignore[invalid-argument-type, not-subscriptable]

    def forward(self, batch):
        doc_embeddings = []
        for doc in batch:
            doc_embeddings.append(self.embed_paper(doc).to(self.device))
        paper_means, paper_variances = self.paper_head(
            torch.cat(doc_embeddings)
        )  # [B, shared_dim]
        return paper_means, paper_variances
