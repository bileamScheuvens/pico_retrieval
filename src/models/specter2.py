import torch
from src.constants import CACHEPATH
from joblib import Memory
from src.models.prob_encoder import ProbabilisticEncoder
from dataclasses import dataclass
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import lightning as L


@dataclass
class SPECTER2Config:
    base_url: str = "allenai/specter2_base"
    adapter: str = "proximity"
    max_len: int = 512
    hidden_dim: int = 256
    shared_dim: int = 128


class SPECTER2Model(L.LightningModule):
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

    def forward(self, batch):
        doc_embeddings = []
        for doc in batch:
            doc_embeddings.append(self.embed_paper(doc).to(self.device))
        return self.prob_encoder(torch.cat(doc_embeddings))  # [B, shared_dim]
