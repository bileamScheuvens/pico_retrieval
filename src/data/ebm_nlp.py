from dataclasses import dataclass
import os
from pathlib import Path
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from src.constants import DATAPATH


def load_ebm():
    docs = {}
    for file in os.scandir(DATAPATH / "ebm-nlp" / "ebm_nlp_2_00" / "documents"):
        file = Path(file)
        if file.suffix != ".txt":
            continue
        with open(file) as f:
            docs[file.stem] = f.read()
    return docs


class SciDocDataset(Dataset):
    def __init__(self):
        super().__init__()

        ebm_nlp = load_ebm()
        self.pmids = list(ebm_nlp.keys())
        self.titles = {}
        self.abstracts = {}
        for pmid, text in ebm_nlp.items():
            title, abstract = text.split("\n\n", 1)
            self.titles[pmid] = title
            self.abstracts[pmid] = abstract

    def __len__(self):
        return len(self.pmids)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        pmid = self.pmids[idx]
        return self.titles[pmid], self.abstracts[pmid]


@dataclass
class SciDocDataConfig:
    batch_size: int = 16
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    seed: int = 161


class SciDocDatamodule(L.LightningDataModule):
    def __init__(self, cfg: SciDocDataConfig):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.cfg = cfg

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data, self.val_data = random_split(
                dataset=SciDocDataset(),
                lengths=[self.cfg.train_ratio, self.cfg.val_ratio],
                generator=torch.Generator().manual_seed(self.cfg.seed),
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
