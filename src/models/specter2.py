from torch import nn
import torch
from adapters import AutoAdapterModel
from joblib import Memory
from transformers import AutoTokenizer

from src.constants import CACHEPATH
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
        self.model = AutoAdapterModel.from_pretrained(cfg.base_url)

        self.model.load_adapter(
            "allenai/specter2_regression",
            source="hf",
            load_as=cfg.adapter,
            set_active=True,
        )
        for p in self.model.parameters():
            p.requires_grad = False

        # cache specter embeddings:
        def _embed_paper(paper):
            inputs = self.tokenizer(
                paper,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=self.cfg.max_len,
            ).to(self.device)
            output = self.model(**inputs)
            # take the first token in the batch as the embedding
            return output.last_hidden_state[:, 0, :]

        self.memory = Memory(CACHEPATH, verbose=0)
        self.embed_paper = self.memory.cache(_embed_paper)

        self.prob_encoder = ProbabilisticEncoder(
            self.model.config.hidden_size,
            cfg.hidden_dim,
            cfg.shared_dim,
        )
        if cfg.use_prob_encoder:
            # turn point embeddings to gaussian
            self.paper_head = ProbabilisticEncoder(
                self.model.config.hidden_size,
                cfg.hidden_dim,
                cfg.shared_dim,
            )
        else:
            self.paper_head = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, cfg.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, cfg.shared_dim),
            )

    @property
    def embed_type(self):
        if self.cfg.use_prob_encoder:
            return "prob"
        return "point"

    def forward(self, batch):
        doc_embeddings = []
        for doc in batch:
            doc_embeddings.append(self.embed_paper(doc).to(self.device))
        paper_means, paper_variances = self.paper_head(
            torch.cat(doc_embeddings)
        )  # [B, shared_dim]
        return paper_means, paper_variances
