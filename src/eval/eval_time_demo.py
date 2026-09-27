from timeit import timeit

import pandas as pd
from omegaconf import DictConfig

from src.data.indexing import eval_index_load, build_ebm_index
from src.models.artsy import ARTSY


def eval_time_demo(cfg: DictConfig):
    model = ARTSY.load_from_checkpoint(
        cfg.model.ckpt_path, weights_only=False, strict=False
    )
    model.eval()
    index, idx2pmid, pmid2content = build_ebm_index(
        model, index_name=cfg.index_name, clear=False
    )
    # index, idx2pmid, pmid2content = eval_index_load(
    #     index_name=cfg.index_name, k_shards=cfg.k_shards, clear=False
    # )

    def embed_query():
        pico = {
            "Patient": ["children with respiratory infection"],
            "Intervention": ["antibiotics"],
            "Control": ["placebo"],
            "Outcome": ["length of hospital"],
        }
        return model.embed_query(pico).numpy()

    def search_index(query_embed, k=15):
        sim_scores, ranks = index.search(query_embed, k)
        return sim_scores, ranks

    def format_results(sim_scores, ranks):
        res = []
        for sim, rank in zip(sim_scores[0], ranks[0]):
            pmid = idx2pmid[rank]
            idx, title, abstract = pmid2content[pmid]
            # model.extract_pico(model.join_text(title, abstract))

            res.append((sim.round(3), pmid, title, abstract))
        return pd.DataFrame(
            data=res,
            columns=["Similarity", "PMID", "Title", "Abstract"],  # ty:ignore[invalid-argument-type]
        )

    query_embed = embed_query()
    sim_scores, ranks = search_index(query_embed)
    n = 1000
    print(
        f"embed query: {timeit('embed_query()', globals=locals(), number=n) / n * 1000}ms",
    )
    print(
        f"search_index: {timeit('search_index(query_embed)', globals=locals(), number=n) / n * 1000}ms"
    )
    print(
        f"format results: {timeit('format_results(sim_scores, ranks)', globals=locals(), number=n) / n * 1000}ms"
    )
