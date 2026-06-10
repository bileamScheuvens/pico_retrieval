import json
from src.utils.configs import SciDocDataConfig
import os
from pathlib import Path
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from src.constants import DATAPATH


def load_pubmed():
    with open(DATAPATH / "pubmed" / "pubmed.json", "r") as f:
        return json.load(f)


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
    def __init__(self, cfg: SciDocDataConfig):
        super().__init__()
        self.cfg = cfg

        self.pmids = []
        self.titles = {}
        self.abstracts = {}

        if cfg.use_ebm_nlp:
            ebm_nlp = load_ebm()
            for pmid, text in ebm_nlp.items():
                title, abstract = text.split("\n\n", 1)
                self.pmids.append(pmid)
                self.titles[pmid] = title
                self.abstracts[pmid] = abstract
        if cfg.use_pubmed_rct:
            pubmed = load_pubmed()
            for pmid, data in pubmed.items():
                if not data["abstract"]:
                    continue
                self.pmids.append(pmid)
                self.titles[pmid] = data["title"]
                self.abstracts[pmid] = data["abstract"]

    def __len__(self):
        return len(self.pmids)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        pmid = self.pmids[idx]
        return self.titles[pmid], self.abstracts[pmid]


class SciDocDatamodule(L.LightningDataModule):
    def __init__(self, cfg: SciDocDataConfig):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.cfg = cfg

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data, self.val_data = random_split(
                dataset=SciDocDataset(self.cfg),
                lengths=[self.cfg.train_ratio, self.cfg.val_ratio],
                generator=torch.Generator().manual_seed(self.cfg.seed),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4)
