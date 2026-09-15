from collections import defaultdict
import json

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.constants import DATAPATH, EXPORTPATH
from src.data.indexing import eval_index_load
from src.models.artsy import ARTSY


def eval_sysrev(cfg: DictConfig, from_seed=False):

    SYSREV_PATH = DATAPATH / "sysrev-seed-collection"
    with (SYSREV_PATH / "overall_collection.jsonl").open(encoding="utf-8") as f:
        sysrev_collection = [json.loads(x) for x in list(f)]

    with (SYSREV_PATH / "pico_search.json").open(encoding="utf-8") as f:
        pico_queries = json.load(f)

    seed_suffix = "seed_" if from_seed else ""

    index, idx2pmid, pmid2content = eval_index_load(cfg.index_name, cfg.k_shards)

    model = ARTSY.load_from_checkpoint(cfg.model.ckpt_path, weights_only=False)
    model.eval()

    def _extract_from_seed(seed_studies):
        seed_pico = defaultdict(lambda: [model.pico_extractor.MISSING_TOKEN])
        for seed in seed_studies:
            if seed not in pmid2content:
                continue
            _, title, abstract = pmid2content[seed]
            seed_pico.update(model.extract_pico(model.join_text(title, abstract)))
        return seed_pico

    def _evaluate_gold(id, included_studies, seed_studies):

        ### create query from combined seed study pico ###
        if from_seed:
            pico = _extract_from_seed(seed_studies)
        ### handwritten query from search title ###
        else:
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
        rows += _evaluate_gold(
            gold["id"], gold["included_studies"], gold["seed_studies"]
        )

    df = pd.DataFrame(rows)

    rcts = df[df["rank"] != -1]
    grouped_df: pd.DataFrame = rcts.groupby(by="sysrev_id")["rank"].agg(
        ranks=list, count="count"
    )
    grouped_df["Total Studies"] = grouped_df["count"] + (
        df[df["rank"] == -1]
        .groupby(by="sysrev_id")["rank"]
        .count()
        .reindex(grouped_df.index, fill_value=0)
    )

    def average_precision(ranks):
        """https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval), adjusted for zero based indexing."""
        AP = 0
        for i, k in enumerate(sorted(ranks)):
            AP += ((i + 1) / (k + 1)) / len(ranks)
        return AP

    grouped_df["AP"] = grouped_df["ranks"].apply(average_precision)
    grouped_df["First Hit"] = grouped_df["ranks"].apply(
        lambda ranks: np.array(ranks).min() + 1
    )
    grouped_df["RR"] = grouped_df["First Hit"].apply(lambda rank: 1 / rank)
    grouped_df["Last Hit"] = grouped_df["ranks"].apply(
        lambda ranks: np.array(ranks).max() + 1
    )

    # wrangling
    grouped_df.index = grouped_df.index.astype("int")
    grouped_df.sort_index(inplace=True)
    grouped_df.reset_index(inplace=True)
    grouped_df.rename(columns={"count": "RCTs", "sysrev_id": "SysRev ID"}, inplace=True)

    grouped_df = grouped_df[
        ["SysRev ID", "RCTs", "Total Studies", "AP", "RR", "First Hit", "Last Hit"]
    ]

    # first and last hit for all studies
    grouped_df.to_csv(
        EXPORTPATH / f"{cfg.index_name}_{seed_suffix}sysrev.csv", index=False
    )

    # table for appendix
    with (EXPORTPATH / f"{cfg.index_name}_{seed_suffix}sysrev.typ").open("w") as f:
        grouped_df.style.hide(axis="index").to_typst(f)

    # raw ranks
    np.savetxt(
        EXPORTPATH / f"{cfg.index_name}_{seed_suffix}sysrev_ranks.csv",
        rcts["rank"].values,
        delimiter=",",
    )

    print(grouped_df)
    print(f"RCTs: {len(rcts)} of {len(df)}")
    print(f"MAP: {sum(grouped_df['AP']) / len(grouped_df):.6f}")
    print(f"MRR: {sum(grouped_df['RR']) / len(grouped_df):.6f}")
