import json
from tqdm import tqdm
from src.constants import CONFIGPATH, EVALPATH
from src.data.indexing import build_eval_index
import hydra
import numpy as np
from omegaconf import DictConfig

from src.models.artsy import ARTSY


@hydra.main(version_base=None, config_path=str(CONFIGPATH), config_name="eval")
def eval_ebm_nlp(cfg: DictConfig):
    # model = ARTSY(cfg.model)
    model = ARTSY.load_from_checkpoint(cfg.model.ckpt_path, weights_only=False)
    index, idx_to_pmid, pmid_to_content = build_eval_index(
        model, index_name=cfg.index_name, clear=False
    )
    all_ranks = []
    for pmid, content in tqdm(pmid_to_content.items(), desc="Performing Evaluation"):
        idx, title, abstract = content
        pico = model.extract_pico(model.join_text(title, abstract))
        pico_embed = model.embed_query(pico)
        # TODO do something with sim
        sim, ranks = index.search(pico_embed, index.ntotal)
        all_ranks.append(np.where(ranks == idx)[1])
    all_ranks = np.array(all_ranks)
    results = {
        "mrr": np.mean(1 / (all_ranks + 1)).round(3),
        "recall@1": np.mean(all_ranks < 1).round(2),
        "recall@10": np.mean(all_ranks < 10).round(2),
        "recall@100": np.mean(all_ranks < 100).round(2),
        "recall@1000": np.mean(all_ranks < 1000).round(2),
    }
    with open(EVALPATH / f"{cfg.index_name}.json", "w") as f:
        f.write(json.dumps(results))

    # pico = {
    #     "Patient": ["subjects without pxs"],
    #     "Intervention": ["Pupillay Dilation"],
    #     "Control": ["Intraocular Pressure and Anterior Segment Morphology"],
    #     "Outcome": [],
    # }


if __name__ == "__main__":
    eval_ebm_nlp()
