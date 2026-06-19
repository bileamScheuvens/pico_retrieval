import os

import faiss
import torch
from tqdm import tqdm

from src.constants import INDEXPATH
from src.data.scidocdata import load_ebm
from src.models.artsy import ARTSY


@torch.no_grad
def build_index(model: ARTSY, index_name, clear=False):
    full_path = INDEXPATH / f"{index_name}.faiss"
    if os.path.exists(full_path) and not clear:
        index = faiss.read_index(str(full_path))
    else:
        index = faiss.IndexFlatIP(model.cfg.paper_embedder.shared_dim)
    ebm_nlp = load_ebm()
    index_size = index.ntotal
    idx_to_pmid = {}
    pmid_to_content = {}
    for i, (pmid, text) in tqdm(
        enumerate(ebm_nlp.items()), desc="Building Index.", total=len(ebm_nlp)
    ):
        title, abstract = text.split("\n\n", 1)
        idx_to_pmid[i] = pmid
        pmid_to_content[pmid] = (i, title, abstract)
        if i < index_size:
            continue
        index.add(model.embed_paper(title, abstract))  # ty:ignore[missing-argument]
    faiss.write_index(index, str(full_path))
    return index, idx_to_pmid, pmid_to_content
