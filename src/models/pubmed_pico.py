from collections import defaultdict

import lightning as L
import torch

from pydantic.dataclasses import dataclass
from joblib import Memory
from sentence_transformers import SentenceTransformer
from seqeval.metrics.sequence_labeling import get_entities

from src.models.prob_encoder import ProbabilisticEncoder

## monkeypatch NERDA
import transformers

transformers.AdamW = torch.optim.AdamW
_strict_load = torch.nn.Module.load_state_dict


def _unstrict_load(self, state_dict, strict=False, **kwargs):
    return _strict_load(self, state_dict, strict=strict, **kwargs)


torch.nn.Module.load_state_dict = _unstrict_load

from NERDA.models import NERDA

from src.constants import MODELPATH, CACHEPATH


@dataclass
class PubMedPicoConfig:
    base_url: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    text_embedder_url: str = "neuml/pubmedbert-base-embeddings"
    max_len: int = 512
    hidden_dim: int = 256
    shared_dim: int = 128


class PubMedPicoModel(L.LightningModule):
    def __init__(self, cfg: PubMedPicoConfig):
        super().__init__()
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

        self.cfg = cfg

        # extract pico from text
        ## load and freeze extractor
        self.extractor = NERDA(
            cfg.base_url, tag_scheme=tag_scheme, tag_outside="O", max_len=cfg.max_len
        )
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
        self.memory = Memory(CACHEPATH, verbose=0)
        self.extract_pico = self.memory.cache(_extract_pico)

        # get point embeddings from pico
        self.text_encoder = SentenceTransformer(cfg.text_embedder_url)

        # turn point embeddings to gaussian
        self.prob_encoder = ProbabilisticEncoder(
            self.text_encoder.get_sentence_embedding_dimension(),
            cfg.hidden_dim,
            cfg.shared_dim,
        )

    def forward(self, text):
        PICO = self.extract_pico(text)

        pico_embeddings = []
        # TODO introduce max number of elements and pad for batching?
        for pico_type in ["Patient", "Intervention", "Control", "Outcome"]:
            elements = PICO[pico_type] or [""]
            for e in elements:
                with torch.no_grad():
                    point_embed = (
                        self.text_encoder.encode(
                            f"{pico_type}: {e}", convert_to_tensor=True
                        )
                        .clone()
                        .to(self.device)
                    )
                prob_embed = self.prob_encoder(point_embed)
                pico_embeddings.append(prob_embed)
        return pico_embeddings
