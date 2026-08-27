from src.models.artsy import ARTSY
from src.data.indexing import eval_index_load
from omegaconf import DictConfig
import json
from src.constants import DATAPATH


def eval_sysrev(cfg: DictConfig):
    with (DATAPATH / "sysrev-seed-collection" / "overall_collection.jsonl").open(
        encoding="utf-8"
    ) as f:
        sysrev_collection = [json.loads(x) for x in list(f)]

    index, idx2pmid, pmid2content = eval_index_load(cfg.index_name, cfg.k_shards)
    model = ARTSY(cfg.model)
    model.eval()

    def _predict(Population, Intervention, Comparator, Outcome, k=15):
        pico = {
            "Patient": Population.split("|"),
            "Intervention": Intervention.split("|"),
            "Control": Comparator.split("|"),
            "Outcome": Outcome.split("|"),
        }
        breakpoint()
        pico_embed = model.embed_query(pico).numpy()
        sim_scores, ranks = index.search(pico_embed, k)
        res = []
        for sim, rank in zip(sim_scores[0], ranks[0]):
            idx, title, abstract = pmid2content[idx2pmid[rank]]
            # modl.extract_pico(model.join_text(title, abstract))

            res.append((sim, rank, title, abstract))
        return res

    _predict("pregnant women", "physical exercise", "", "gestational diabetes", 5)

    pass
