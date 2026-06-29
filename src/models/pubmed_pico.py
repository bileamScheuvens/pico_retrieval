from collections import defaultdict

import torch
import torch.nn.functional as F

from joblib import Memory
from seqeval.metrics.sequence_labeling import get_entities
from torch import nn

from src.constants import CACHEPATH, MODELPATH
from src.models import PicoExtractor, PicoProjector
from src.models.model_heads import ProbabilisticEncoder, PointEncoder
from src.models.text_embedders import PromptRepsModel, SentenceTransformerModel
from src.utils.configs import PubMedPicoConfig, TextEmbedType

## monkeypatch NERDA
import transformers

transformers.AdamW = torch.optim.AdamW  # ty:ignore[unresolved-attribute]
_strict_load = torch.nn.Module.load_state_dict


def _unstrict_load(self, state_dict, strict=False, **kwargs):
    return _strict_load(self, state_dict, strict=strict, **kwargs)


torch.nn.Module.load_state_dict = _unstrict_load  # ty:ignore[invalid-assignment]

from NERDA.models import NERDA  # noqa: E402


class PubMedPicoProjector(PicoProjector):
    def __init__(self, cfg: PubMedPicoConfig, in_dim: int):
        super().__init__(cfg)

        if cfg.use_prob_encoder:
            self.model = ProbabilisticEncoder(in_dim, cfg.hidden_dim, cfg.shared_dim)
        else:
            self.model = PointEncoder(in_dim, cfg.hidden_dim, cfg.shared_dim)

    def forward(self, pico_embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        return list(self.model(torch.stack(pico_embeddings)))

    @property
    def embed_type(self) -> str:
        if self.cfg.use_prob_encoder:
            return "prob"
        return "point"


class PubMedPicoModel(PicoExtractor):
    def __init__(self, cfg: PubMedPicoConfig):
        super().__init__(cfg)
        tag_scheme = [
            "B-Patient",
            "I-Patient",
            "B-Intervention",
            "I-Intervention",
            "B-Control",
            "I-Control",
            "B-Outcome",
            "I-Outcome",
        ]

        # extract pico from text
        ## load and freeze extractor
        self.extractor = NERDA(
            cfg.base_url, tag_scheme=tag_scheme, tag_outside="O", max_len=cfg.max_len
        )
        self.extractor.transformer_tokenizer
        self.extractor.load_network_from_file(MODELPATH / "pubmedpico.bin")

        for p in self.extractor.network.parameters():
            p.requires_grad = False

        def _extract_pico(text):
            """Annotate pico elements and extract as sequences."""
            words_by_sent, tags_by_sent = self.extractor.predict_text(text)
            PICO = defaultdict(list)
            for words, tags in zip(words_by_sent, tags_by_sent):
                for label, start, end in get_entities(tags):
                    PICO[label].append(" ".join(words[start : end + 1]))
            return PICO

        # cache extractions
        if cfg.cache_extraction:
            self.memory = Memory(CACHEPATH, verbose=0)
            self.extract_pico = self.memory.cache(_extract_pico)
        else:
            self.extract_pico = _extract_pico

        # get point embeddings from pico
        if cfg.text_embed_type == TextEmbedType.SENTENCE:
            self.text_encoder = SentenceTransformerModel(cfg.text_embedder_url)
        elif cfg.text_embed_type == TextEmbedType.PROMPTREPS:
            self.text_encoder = PromptRepsModel(cfg.text_embedder_url)

        if cfg.use_prob_encoder:
            # turn point embeddings to gaussian
            self.pico_head = ProbabilisticEncoder(
                self.text_encoder.embed_dim,
                cfg.hidden_dim,
                cfg.shared_dim,
            )
        else:
            self.pico_head = nn.Sequential(
                nn.Linear(self.text_encoder.embed_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, cfg.shared_dim),
            )

    @property
    def embed_dim(self):
        return self.text_encoder.embed_dim

    def join_text(self, title, abstract):
        return title + self.extractor.transformer_tokenizer.sep_token + abstract  # ty:ignore[unresolved-attribute]

    def forward(
        self,
        text,
    ) -> tuple[list[torch.Tensor], list[str]]:
        PICO = self.extract_pico(text)
        return self.encode_pico(PICO)

    def encode_pico(self, PICO) -> tuple[list[torch.Tensor], list[str]]:
        pico_dropout_coeff = self.cfg.pico_dropout if self.training else 0.0
        pico_embeddings = []
        pico_labels = []
        # TODO introduce max number of elements and pad for batching?
        for pico_type in ["Patient", "Intervention", "Control", "Outcome"]:
            if pico_type not in self.cfg.considered_elements:
                continue
            elements = PICO[pico_type]
            if not elements or (torch.rand(1) < pico_dropout_coeff):
                elements = ["[MISSING]"]
            for e in elements:
                pico_labels.append(pico_type)
                pico_embeddings.append(self.text_encoder(f"{pico_type}: {e}"))
        return pico_embeddings, pico_labels
