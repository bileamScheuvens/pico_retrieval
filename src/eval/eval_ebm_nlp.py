import torch
from tqdm import tqdm
from src.constants import CONFIGPATH
from src.eval import build_index
import hydra
import numpy as np
from omegaconf import DictConfig

from src.models.artsy import ARTSY


@hydra.main(version_base=None, config_path=str(CONFIGPATH), config_name="eval")
def eval_ebm_nlp(cfg: DictConfig):
    # model = ARTSY(cfg.model)
    breakpoint()
    model = ARTSY.load_from_checkpoint(cfg.model.ckpt_path, weights_only=False)
    index, idx_to_pmid, pmid_to_content = build_index(
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
    breakpoint()

    # pico = {
    #     "Patient": ["subjects without pxs"],
    #     "Intervention": ["Pupillay Dilation"],
    #     "Control": ["Intraocular Pressure and Anterior Segment Morphology"],
    #     "Outcome": [],
    # }


if __name__ == "__main__":
    eval_ebm_nlp()
