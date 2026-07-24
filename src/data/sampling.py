from __future__ import annotations

from random import choice
from typing import TYPE_CHECKING

from torch.utils.data import BatchSampler, Sampler, Subset, RandomSampler

from src.data.indexing import PICOIndex

if TYPE_CHECKING:
    from src.data.scidocdata import SciDocDataset
from src.utils.configs import HardNegativeConfig, SamplerType


def SamplerFactory(cfg, dataset):
    if cfg.sampler.sampler_type == SamplerType.RANDOM:
        return BatchSampler(
            RandomSampler(dataset), batch_size=cfg.batch_size, drop_last=False
        )
    if cfg.sampler.sampler_type == SamplerType.HARDNEGATIVE:
        return HardNegativeSampler(cfg.sampler, cfg.batch_size, dataset)

    raise NotImplementedError(f"No sampler of type {cfg.text_embed_type}")


class HardNegativeSampler(Sampler):
    def __init__(self, cfg: HardNegativeConfig, batch_size: int, subset: Subset):
        self.cfg = cfg
        self.n_anchors = cfg.n_anchors
        self.n_per_anchor = cfg.n_per_anchor
        self.batch_size = batch_size
        self.index = PICOIndex(
            model_name=cfg.extractor_model,
            corpus=cfg.corpus,
            n_candidates=cfg.n_candidates,
            n_subcandidates=cfg.n_subcandidates,
        )
        self.subset = subset
        self.subset_idx_map = {idx: i for i, idx in enumerate(subset.indices)}
        self.dataset: SciDocDataset = subset.dataset  # ty:ignore[invalid-assignment]

    def __len__(self):
        return len(self.subset) // self.batch_size

    def __iter__(self):
        """One Batch consists of n_a anchors, n_n negatives per anchor and batch_size-(n_a * (n_n+1)) random samples"""
        for _ in range(len(self)):
            global_dataset_idx = set()
            # gather anchor and hard negatives
            for _ in range(self.n_anchors):
                anchor = self.dataset.pmids[choice(self.subset.indices)]
                global_dataset_idx.add(self.dataset.pmid2idx[anchor])
                # get negative candidates
                negatives = [
                    self.dataset.pmid2idx[x] for x in self.index.get_similar_doc(anchor)
                ]
                # filter out negatives not in this subset
                # TODO find out if this is expensive
                negatives = [x for x in negatives if x in self.subset_idx_map]
                # add to batch
                global_dataset_idx |= set(negatives[: self.n_per_anchor])
            # fill batch with random negatives ensuring no duplicates
            while len(global_dataset_idx) < self.batch_size:
                global_dataset_idx.add(choice(self.subset.indices))
            yield [self.subset_idx_map[x] for x in global_dataset_idx]
