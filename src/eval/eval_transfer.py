import torch
from omegaconf import DictConfig
from src.models.artsy import ARTSY
from src.submodules import SciRepEval, SciRepModel

from src.constants import ROOT, CACHEPATH


class SciRepModelWrapped(SciRepModel):
    def __init__(self, cfg):
        super().__init__(base_checkpoint="allenai/specter")
        self.model = ARTSY.load_from_checkpoint(
            cfg.model.ckpt_path, weights_only=False, strict=False
        )

    def __call__(self, batch, batch_ids=None):
        batch = [batch] if type(batch) == str else batch
        embeddings = []
        for x in batch:
            try:
                title, abstract = x.split("[SEP]")
            except Exception as e:
                continue
            embeddings.append(self.model.embed_paper(title, abstract))

        return torch.cat(embeddings)


def eval_transfer(cfg: DictConfig):
    specter = SciRepModel(base_checkpoint="allenai/specter")
    artsy = SciRepModelWrapped(cfg)

    task_file = ROOT / "src" / "submodules" / "scirepeval" / "scirepeval_tasks.jsonl"
    evaluator = SciRepEval(
        tasks_config=str(task_file),
        task_list=["Biomimicry", "DRSM"],
        embedding_save_path=CACHEPATH / "transfer.pt",
    )

    evaluator.evaluate(artsy, f"transfer_{cfg.index_name}.json")
    for name, model in zip(["artsy", "specter"], [artsy, specter]):
        evaluator.evaluate(model, f"transfer_{name}.json")
