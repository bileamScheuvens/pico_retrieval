import json

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.constants import DATAPATH, EXPORTPATH
from src.data.indexing import eval_index_load
from src.models.artsy import ARTSY


def eval_sysrev(cfg: DictConfig):

    SYSREV_PATH = DATAPATH / "sysrev-seed-collection"
    with (SYSREV_PATH / "overall_collection.jsonl").open(encoding="utf-8") as f:
        sysrev_collection = [json.loads(x) for x in list(f)]

    with (SYSREV_PATH / "pico_search.json").open(encoding="utf-8") as f:
        pico_queries = json.load(f)

    index, idx2pmid, pmid2content = eval_index_load(cfg.index_name, cfg.k_shards)
    model = ARTSY.load_from_checkpoint(cfg.model.ckpt_path, weights_only=False)
    model.eval()

    def _evaluate_gold(id, included_studies):
        pico = {}
        for category in ["Population", "Intervention", "Comparator", "Outcome"]:
            pico[category] = pico_queries[id][category].split("|")

        # handle extractor specific vocab
        pico["Patient"] = pico["Population"]
        pico["Control"] = pico["Comparator"]

        pico_embed = model.embed_query(pico).numpy()
        sim, ranks = index.search(pico_embed, index.ntotal)

        rows = []
        for study in included_studies:
            if study not in pmid2content:
                rows.append({"sysrev_id": id, "pmid": study, "rank": -1})
                continue
            rank = np.where(ranks == pmid2content[study][0])[1]
            rows.append({"sysrev_id": id, "pmid": study, "rank": rank.item()})
        return rows

    rows = []
    for gold in sysrev_collection:
        rows += _evaluate_gold(gold["id"], gold["included_studies"])

    df = pd.DataFrame(rows)

    rcts = df[df["rank"] != -1]
    grouped_df: pd.DataFrame = rcts.groupby(by="sysrev_id")["rank"].agg(
        ranks=list, count="count"
    )

    def average_precision(ranks):
        """https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)"""
        AP = 0
        for i, k in enumerate(sorted(ranks)):
            AP += (i / k) / len(ranks)
        return AP

    grouped_df["AP"] = grouped_df["ranks"].apply(average_precision)
    grouped_df["MRR"] = grouped_df["ranks"].apply(
        lambda ranks: sum(1 / np.array(ranks)) / len(ranks)
    )

    with (EXPORTPATH / f"{cfg.index_name}_sysrev.typ").open("w") as f:
        grouped_df.drop(columns="ranks").style.to_typst(f)
    print(grouped_df)
    print(f"RCTs: {len(rcts)} of {len(df)}")
    print(f"MAP: {sum(grouped_df['AP']) / len(grouped_df):.6f}")
