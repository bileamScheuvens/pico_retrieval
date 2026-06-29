from src.constants import MODELPATH
import torch

import onnxruntime as ort
from transformers import AutoTokenizer

from src.models import PaperEmbedder
from src.models.model_heads import ProbabilisticEncoder, PointEncoder
from src.utils.configs import SPECTER2Config


class SPECTER2Model(PaperEmbedder):
    def __init__(self, cfg: SPECTER2Config):
        super().__init__(cfg)
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_url)

        # load base model and freeze
        self.specter2_session = ort.InferenceSession(
            MODELPATH / f"specter2_{cfg.adapter}.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            self.paper_head = PointEncoder(
                self.specter_out_dim,
                cfg.hidden_dim,
                cfg.shared_dim,
            )

    def join_text(self, title, abstract):
        return title + self.tokenizer.sep_token + abstract  # ty:ignore[unresolved-attribute]

    @property
    def embed_type(self):
        if self.cfg.use_prob_encoder:
            return "prob"
        return "point"

    def embed_batch(self, batch):
        inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="np",
            return_token_type_ids=False,
            max_length=self.cfg.max_len,
        ).to(self.device)  # ty:ignore[call-non-callable]

        output = self.specter2_session.run(["last_hidden_state"], inputs)[0]

        # take the first token as the embedding
        return torch.tensor(output[:, 0, :], device=self.device)  # ty:ignore[invalid-argument-type, not-subscriptable]

    def forward(self, batch):
        doc_embeddings = self.embed_batch(batch)
        return self.paper_head(doc_embeddings)  # [B, shared_dim]
