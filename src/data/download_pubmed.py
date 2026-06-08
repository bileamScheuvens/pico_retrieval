import json
from src.constants import DATAPATH
import time
import os
from Bio import Entrez, Medline
from tqdm import tqdm

Entrez.email = os.getenv("NCBI_EMAIL")  # ty:ignore[invalid-assignment]
Entrez.api_key = os.getenv("NCBI_API_KEY")  # ty:ignore[invalid-assignment]


def download_pubmed():
    dest_path = DATAPATH / "pubmed" / "pubmed.json"
    if os.path.exists(dest_path):
        raise FileExistsError()

    def _refresh_search():
        handle = Entrez.esearch(
            db="pubmed",
            term="randomized controlled trial[Publication Type]",
            retmax=0,
            usehistory="y",
        )
        return Entrez.read(handle)

    query_result = _refresh_search()
    count = int(query_result["Count"])

    def fetch_batch(date):
        ids = Entrez.read(
            Entrez.esearch(
                db="pubmed",
                term=f"randomized controlled trial[Publication Type] AND ({date})",
                retmax=9999,
            )
        )["IdList"]
        with Entrez.efetch(
            db="pubmed",
            id=",".join(ids),
            retmode="text",
            rettype="medline",
        ) as handle:
            for article in Medline.parse(handle):
                articles[article.get("PMID")] = {
                    "title": article.get("TI"),
                    "abstract": article.get("AB"),
                }
            time.sleep(0.1)

    # can download at most 10k at once.
    # start up to 1975, then monthly
    articles = {}
    batches = ["0001[dp]:1975[dp]"]
    for year in range(1976, 2026):
        for month in range(12):
            batches.append(f"{year}/{month}[dp]")
    for date in tqdm(batches, desc="Downloading abstracts"):
        try:
            fetch_batch(date)
        except Exception as e:
            print(f"bad batch, skipped date {date}: {e}")
    with open(dest_path, "w") as f:
        f.write(json.dumps(articles))

    print("Total records:", count)


if __name__ == "__main__":
    download_pubmed()
