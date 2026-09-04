import torch

from src.models import PaperEmbedder, TextEncoderFactory
from src.models.model_heads import PointEncoder, ProbabilisticEncoder
from src.utils.configs import PaperEmbedderConfig


class PlainEmbedder(PaperEmbedder):
    def __init__(self, cfg: PaperEmbedderConfig):
        super().__init__(cfg)
        self.text_encoder = TextEncoderFactory(cfg)

        if cfg.use_prob_encoder:
            # turn point embeddings to gaussian
            self.pico_head = ProbabilisticEncoder(
                self.text_encoder.embed_dim,
                cfg.hidden_dim,
                cfg.shared_dim,
            )
        else:
            self.pico_head = PointEncoder(
                self.text_encoder.embed_dim, cfg.hidden_dim, cfg.shared_dim
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

        return self.paper_head(torch.cat(doc_embeddings))  # [B, shared_dim]
