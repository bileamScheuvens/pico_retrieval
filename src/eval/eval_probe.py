import os
import pandas as pd
import gradio as gr
from src.data.indexing import build_eval_index
from collections import defaultdict

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.constants import EVALPATH
from src.data.scidocdata import SciDocDatamodule
from src.metrics.plots import plot_means_subsets
from src.models.artsy import ARTSY


def eval_probe(cfg: DictConfig):
    model = ARTSY.load_from_checkpoint(
        cfg.model.ckpt_path, weights_only=False, strict=False
    )
    index, idx2pmid, pmid2content = build_eval_index(
        model, index_name=cfg.index_name, clear=False
    )

    def _predict(Population, Intervention, Comparator, Outcome, k=15) -> pd.DataFrame:
        pico = {
            "Patient": Population.split("|"),
            "Intervention": Intervention.split("|"),
            "Control": Comparator.split("|"),
            "Outcome": Outcome.split("|"),
        }
        pico_embed = model.embed_query(pico).numpy()
        sim_scores, ranks = index.search(pico_embed, k)
        res = []
        for sim, rank in zip(sim_scores[0], ranks[0]):
            idx, title, abstract = pmid2content[idx2pmid[rank]]
            # model.extract_pico(model.join_text(title, abstract))

            res.append((sim, rank, title, abstract))
        return pd.DataFrame(
            data=res,
            columns=["Similarity", "PMID", "Title", "Abstract"],  # ty:ignore[invalid-argument-type]
        )

    demo = gr.Interface(
        fn=_predict,
        inputs=["text", "text", "text", "text"],
        outputs=gr.DataFrame(wrap=True),
    )
    demo.launch(share=True)
