import json
import os
from collections import defaultdict

import faiss
import numpy as np
import torch
from tqdm import tqdm

from src.constants import INDEXPATH
from src.data.scidocdata import load_ebm
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
        index.add(model.embed_paper(title, abstract))  # ty:ignore[missing-argument]
        if i % 2500 == 0 or i == len(ebm_nlp) - 1:
            faiss.write_index(index, str(full_path))
    return index, idx_to_pmid, pmid_to_content


class PICOIndex:
    """
    Index over all extracted pico elements, embedded with a language model. Implements set similarity:
    sim(docA, docB) = mean(max_wrt_P(sim(docA, docB)), ...)
    """

    def __init__(self, model: PicoExtractor, n_candidates=50, n_subcandidates=50):
        self.model = model
        self.dim = model.embed_dim
        self.n_candidates, self.n_subcandidates = n_candidates, n_subcandidates
        self.pmid2content = {}
        self.index = {}
        self.idx2pmid = {}
        self.pmid2idx = {}
        for pico_category in ["Patient", "Intervention", "Control", "Outcome"]:
            index, idx2pmid, pmid2idx = self.get_subindex(pico_category)
            self.index[pico_category] = index
            self.idx2pmid[pico_category] = idx2pmid
            self.pmid2idx[pico_category] = pmid2idx

        self.populate()

    def get_subindex_path(self, category):
        return INDEXPATH / f"{category}.faiss"

    def get_legend_paths(self, category):
        return INDEXPATH / f"{category}.json", INDEXPATH / f"{category}_inv.json"

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
        ebm_nlp = load_ebm()
        for i, (pmid, text) in tqdm(
            enumerate(ebm_nlp.items()), desc="Building Index.", total=len(ebm_nlp)
        ):
            if pmid in self.pmid2content:
                continue
            title, abstract = text.split("\n\n", 1)
            self.pmid2content[pmid] = (title, abstract)
            text = self.model.join_text(title, abstract)  # ty:ignore[call-non-callable]
            # TODO check if keys are really
            for vec, label in zip(*self.model(text)):
                self.add_sample(vec, pmid, label)
            if i % 2500 == 0 or i == len(ebm_nlp) - 1:
                self.write_indices()

    def add_sample(self, vec, pmid, pico_category):
        i = self.index[pico_category].ntotal
        self.idx2pmid[pico_category][i] = pmid
        self.pmid2idx[pico_category][pmid].append(i)
        self.index[pico_category].add(vec)

    def write_indices(self):
        for category, index in self.index.items():
            legend_path, legend_inv_path = self.get_legend_paths(category)
            with legend_path.open() as f:
                json.dump(self.idx2pmid[category], f)
            with legend_inv_path.open() as f:
                json.dump(self.pmid2idx[category], f)
            faiss.write_index(index, str(self.get_subindex_path(category)))

    def get_similar_doc(self, anchor_pmid):
        """Get similar doc by comparing picos."""

        candidates = defaultdict(float)

        # Loop over  P I C O
        for pico_category in ["Patient", "Intervention", "Control", "Outcome"]:
            candidates_per_category = defaultdict(float)
            elements = self.pmid2idx[anchor_pmid]
            # For each extracted P, I, C, O
            for cur_pico_idx in elements:
                # search for n_subcandidates most similar ones and aggregate similarities
                cur_pico_vec = self.index[pico_category].reconstruct(cur_pico_idx)
                sim, doc_idx = self.index[pico_category].search(
                    cur_pico_vec, self.n_subcandidates
                )
                candidates_per_category[self.idx2pmid[doc_idx]] += sim
            # transfer mean sim for each doc to global_candidates
            for c, sim in candidates_per_category.items():
                candidates[c] += sim / len(elements)
        # prevent retrieving itself
        candidates[anchor_pmid] = 0
        # return top candidates docs
        # TODO check ascending correct
        return np.array(candidates.values()).sort()[: self.n_candidates]
