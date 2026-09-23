import gradio as gr
import pandas as pd
from omegaconf import DictConfig

from src.data.indexing import build_ebm_index, eval_index_load
from src.models.artsy import ARTSY


def eval_probe(cfg: DictConfig):
    model = ARTSY.load_from_checkpoint(
        cfg.model.ckpt_path, weights_only=False, strict=False
    )
    model.eval()
    index, idx2pmid, pmid2content = eval_index_load(
        index_name=cfg.index_name, k_shards=cfg.k_shards, clear=False
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
            pmid = idx2pmid[rank]
            idx, title, abstract = pmid2content[pmid]
            # model.extract_pico(model.join_text(title, abstract))

            res.append((sim.round(3), rank, title, abstract))
        return pd.DataFrame(
            data=res,
            columns=["Similarity", "PMID", "Title", "Abstract"],  # ty:ignore[invalid-argument-type]
        )

    with gr.Blocks() as demo:
        gr.Markdown("## CoPPeR inference")

        with gr.Column():
            p = gr.Textbox(label="Population")
            i = gr.Textbox(label="Intervention")
            c = gr.Textbox(label="Comparator")
            o = gr.Textbox(label="Outcome")
            submit = gr.Button("Search", variant="primary")
            df = gr.DataFrame(wrap=False, label="Results")
        submit.click(fn=_predict, inputs=[p, i, c, o], outputs=df)

    demo.launch(share=True)
