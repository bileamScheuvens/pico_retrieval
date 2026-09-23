import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

import wandb
from src.constants import EXPORTPATH

api = wandb.Api()


@dataclass
class Filter:
    table: str
    group_name: str | None = None
    columns: list | None = None
    name_transform: Callable | None = None
    generic_transform: Callable | None = None


def get_runs(filter):
    runs = api.runs(
        "benedictbileam/pico_retrieval", filters={"group": filter.group_name}
    )

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)

        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    df = pd.json_normalize(summary_list).round(3)
    df["name"] = name_list
    if filter.name_transform:
        df["name"] = df["name"].apply(filter.name_transform)
    if filter.generic_transform:
        df = filter.generic_transform(df)
    if filter.columns:
        df = df[[*filter.columns]]
    df.to_csv(EXPORTPATH / "wandb" / f"{filter.table}.csv", index=False)


def split_comp_name(df):
    def _extract_ops(x):
        # split name into parts [..., inter, picoaggtype, intra]
        split = re.split(r"_|\.", x)
        return split[-1][:5], split[-3][:5]

    df["inter"], df["intra"] = zip(*df["name"].apply(_extract_ops))
    df.dropna(inplace=True)
    return df


filters = [
    # Filter(
    #     "lr",
    #     group_name="sweep_lr",
    #     columns=["name", "val/loss", "val/recall"],
    #     name_transform=lambda x: f"{float(x[9:]):.1e}".replace("e-0", "e^(-") + ")",
    # )
    # Filter(
    #     "embed_dim",
    #     group_name="sweep_shared_dim",
    #     columns=["name", "val/loss", "val/recall"],
    #     name_transform=lambda x: x.split("-")[1],
    # )
    # Filter(
    #     "compositors",
    #     group_name="sweep_intra_agg_inter_agg",
    #     columns=["inter", "intra", "val/loss", "val/recall"],
    #     generic_transform=split_comp_name,
    # )
]

if __name__ == "__main__":
    for x in filters:
        get_runs(x)
