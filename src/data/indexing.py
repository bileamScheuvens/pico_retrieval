import json
import os
from collections import defaultdict
from itertools import chain
from typing import Optional

import faiss
import torch
from tqdm import tqdm

from src.constants import INDEXPATH
from src.data.load_data import ebm_iter, load_ebm, pubmed_iter
from src.models import PicoExtractor
from src.models.artsy import ARTSY


@torch.no_grad
def build_eval_index(model: ARTSY, index_name, clear=False):
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
        index.add(model.embed_paper(title, abstract).numpy())  # ty:ignore[missing-argument]
        if i % 2500 == 0 or i == len(ebm_nlp) - 1:
            faiss.write_index(index, str(full_path))
    return index, idx_to_pmid, pmid_to_content


class PICOIndex:
    """
    Index over all extracted pico elements, embedded with a language model. Implements set similarity:
    sim(docA, docB) = sum(mean(sim(docA_P, docB_P)), sim(docA_I, docB_I),...)
    """

    def __init__(
        self,
        extractor: Optional[PicoExtractor] = None,
        model_name: Optional[str] = None,
        corpus="both",
        n_candidates=50,
        n_subcandidates=50,
    ):
        if extractor is not None:
            self.extractor = extractor
            self.model_name = extractor.cfg.text_embedder.name
            self.dim = extractor.embed_dim
        else:
            if model_name is None:
                raise ValueError("If no extractor is passed needs model_name.")
            self.model_name = model_name
        self.corpus = corpus
        self.n_candidates, self.n_subcandidates = n_candidates, n_subcandidates

        # load content map
        if os.path.exists(self.content_map_path):
            self.pmid2content = json.load(self.content_map_path.open())
        else:
            self.pmid2content = {}

        self.index = {}
        self.idx2pmid = {}
        self.pmid2idx = {}
        # load index and index maps
        for pico_category in ["Patient", "Intervention", "Control", "Outcome"]:
            index, idx2pmid, pmid2idx = self.get_subindex(pico_category)
            self.index[pico_category] = index
            self.idx2pmid[pico_category] = idx2pmid
            self.pmid2idx[pico_category] = pmid2idx

        self.populate()

    @property
    def content_map_path(self):
        return INDEXPATH / f"meta_{self.corpus}_{self.model_name}.json"

    def get_subindex_path(self, category):
        return INDEXPATH / f"{category}_{self.corpus}_{self.model_name}.faiss"

    def get_legend_paths(self, category):
        return (
            INDEXPATH / f"{category}_{self.corpus}_{self.model_name}.json",
            INDEXPATH / f"{category}_{self.corpus}_{self.model_name}_inv.json",
        )

    def get_subindex(self, category):
        idx_path = self.get_subindex_path(category)
        legend_path, legend_inv_path = self.get_legend_paths(category)
        if os.path.exists(idx_path):
            index = faiss.read_index(str(idx_path))
            idx_to_pmid = json.load(legend_path.open())
            pmid_to_idx = defaultdict(list, json.load(legend_inv_path.open()))
        else:
            index = faiss.IndexFlatIP(self.dim)
            idx_to_pmid = {}
            pmid_to_idx = defaultdict(list)
        return index, idx_to_pmid, pmid_to_idx

    @torch.no_grad
    def populate(self):
        if self.corpus == "ebm_nlp":
            data, data_len = ebm_iter()
        elif self.corpus == "pubmed":
            data, data_len = pubmed_iter()
        elif self.corpus == "both":
            ebm, ebm_len = ebm_iter()
            pubmed, pubmed_len = pubmed_iter()
            data = chain(ebm, pubmed)
            data_len = pubmed_len + ebm_len  # slightly wrong due to overlap

        modified_index = False
        for i, (pmid, title, abstract) in tqdm(
            enumerate(data), desc="Building Index.", total=data_len
        ):
            if pmid in self.pmid2content:
                continue
            self.pmid2content[pmid] = (title, abstract)
            modified_index = True
            if not hasattr(self, "extractor"):
                raise ValueError("Incomplete Index but no extractor model provided.")
            text = self.extractor.join_text(title, abstract)  # ty:ignore[call-non-callable]
            # TODO check if keys are really
            for vec, label in zip(*self.extractor(text)):
                self.add_sample(vec, pmid, label)
            if i % 500 == 0 or i == data_len - 1:
                self.write_indices()

        if modified_index:
            self.write_indices()

    def add_sample(self, vec, pmid, pico_category):
        i = self.index[pico_category].ntotal
        self.idx2pmid[pico_category][i] = pmid
        self.pmid2idx[pico_category][pmid].append(i)
        self.index[pico_category].add(vec.unsqueeze(0))

    def write_indices(self):
        with self.content_map_path.open("w") as f:
            json.dump(self.pmid2content, f)
        for category, index in self.index.items():
            legend_path, legend_inv_path = self.get_legend_paths(category)
            with legend_path.open("w") as f:
                json.dump(self.idx2pmid[category], f)
            with legend_inv_path.open("w") as f:
                json.dump(self.pmid2idx[category], f)
            faiss.write_index(index, str(self.get_subindex_path(category)))

    def get_similar_doc(self, anchor_pmid):
        """Get similar doc by comparing picos."""

        candidates = defaultdict(float)

        # Loop over P I C O
        for pico_category in ["Patient", "Intervention", "Control", "Outcome"]:
            candidates_per_category = defaultdict(float)
            elements = self.pmid2idx[pico_category][anchor_pmid]
            # For each extracted P, I, C, O
            for cur_pico_idx in elements:
                # search for n_subcandidates most similar ones and aggregate similarities
                cur_pico_vec = self.index[pico_category].reconstruct(cur_pico_idx)
                sims, doc_idxs = self.index[pico_category].search(
                    cur_pico_vec[None, ...], self.n_subcandidates
                )
                for sim, doc_idx in zip(sims[0], doc_idxs[0]):
                    candidates_per_category[
                        self.idx2pmid[pico_category][str(doc_idx)]
                    ] += sim
            # transfer mean sim for each doc to global_candidates
            for c, sim in candidates_per_category.items():
                candidates[c] += sim / len(elements)
        # prevent retrieving itself
        candidates[anchor_pmid] = 0
        # return top candidates docs
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return list(dict(sorted_candidates[: self.n_candidates]).keys())
