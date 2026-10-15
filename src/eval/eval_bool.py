import pandas as pd
import json

from ..constants import DATAPATH

SYSREV_PATH = DATAPATH / "sysrev-seed-collection"

if __name__ == "__main__":
    bool_query_results = pd.read_csv(
        DATAPATH / "sysrev-seed-collection" / "candidate_documents.res",
        names=["SysRev ID", "Boolean Pool", "pmid", "2", "3", "4"],
        delimiter=" ",
    )
    bool_query_results = bool_query_results.groupby("SysRev ID").count()["Boolean Pool"]

    sysrev_collection = pd.read_json(
        SYSREV_PATH / "overall_collection.jsonl", lines=True
    )
    sysrev_collection["TP"] = sysrev_collection["included_studies"].apply(len)
    res = sysrev_collection[["id", "TP"]].set_index("id")
    res = res.join(bool_query_results)
    print((res["TP"] / res["Boolean Pool"]).mean())
