import json
import os
from pathlib import Path

from src.constants import DATAPATH


def load_pubmed():
    with open(DATAPATH / "pubmed" / "pubmed.json", "r") as f:
        return json.load(f)


def pubmed_iter():
    data = load_pubmed()

    def _pubmed_iter():
        for pmid, doc in data.items():
            if not doc["abstract"]:
                continue
            yield pmid, doc["title"], doc["abstract"]

    return _pubmed_iter(), len(data)


def load_ebm():
    docs = {}
    for file in os.scandir(DATAPATH / "ebm-nlp" / "ebm_nlp_2_00" / "documents"):
        file = Path(file)
        if file.suffix != ".txt":
            continue
        with open(file) as f:
            docs[file.stem] = f.read()
    return docs


def ebm_iter():
    data = load_ebm()

    def _ebm_iter():
        for pmid, doc in data.items():
            title, abstract = doc.split("\n\n", 1)
            yield pmid, title, abstract

    return _ebm_iter(), len(data)
