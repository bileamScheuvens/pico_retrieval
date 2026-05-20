## monkeypatch NERDA
from collections import defaultdict
import transformers
from types import MethodType
import torch
from seqeval.metrics.sequence_labeling import get_entities

transformers.AdamW = torch.optim.AdamW
_strict_load = torch.nn.Module.load_state_dict


def _unstrict_load(self, state_dict, strict=False, **kwargs):
    return _strict_load(self, state_dict, strict=strict, **kwargs)


torch.nn.Module.load_state_dict = _unstrict_load


from NERDA.models import NERDA
from src.constants import MODELPATH


class PubMedPicoModel:
    def __init__(self):
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

        base_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

        self.model = NERDA(
            base_model, tag_scheme=tag_scheme, tag_outside="O", max_len=512
        )
        self.model.load_network_from_file(MODELPATH / "pubmedpico.bin")

    def predict_pico(self, text):
        words_by_sent, tags_by_sent = self.model.predict_text(text)
        PICO = defaultdict(list)
        for words, tags in zip(words_by_sent, tags_by_sent):
            for label, start, end in get_entities(tags):
                PICO[label].append(" ".join(words[start : end + 1]))
        return PICO
