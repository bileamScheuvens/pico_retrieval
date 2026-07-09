from dash import Dash, html, dcc, callback, Output, Input
from collections import defaultdict

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.scidocdata import SciDocDatamodule
from src.metrics.plots import plot_means_dash
from src.models.artsy import ARTSY

pico_subsets = {
    "PICO": ["Patient", "Intervention", "Control", "Outcome"],
    "PIC": ["Patient", "Intervention", "Control"],
    "PIO": ["Patient", "Intervention", "Outcome"],
    "PCO": ["Patient", "Control", "Outcome"],
    "ICO": ["Intervention", "Control", "Outcome"],
}


def eval_dash(cfg: DictConfig):
    app = Dash()
    app.layout = [
        dcc.Dropdown(
            ["Patient", "Intervention", "Control", "Outcome"],
            searchable=False,
            multi=True,
            placeholder="Paper",
            id="selection_A",
        ),
        dcc.Dropdown(
            ["Patient", "Intervention", "Control", "Outcome"],
            searchable=False,
            multi=True,
            placeholder="Paper",
            id="selection_B",
        ),
        dcc.Graph(id="fig", style={"width": "90vh", "height": "90vh"}),
    ]

    model = ARTSY.load_from_checkpoint(
        cfg.model.ckpt_path, weights_only=False, strict=False
    )
    model.eval()
    datamodule = SciDocDatamodule(cfg.data)
    datamodule.setup("fit")
    val_data = datamodule.val_data
    N = 128
    titles = []
    paper_means = []
    picos = []
    # precompute paper embed and pico extraction
    for i in tqdm(range(N)):
        title, abstract = val_data[i]
        titles.append(title)
        # TODO treat prob
        paper_means.append(model.embed_paper(title, abstract))
        picos.append(model.extract_pico(model.join_text(title, abstract)))

    @callback(
        Output("fig", "figure"),
        Input("selection_A", "value"),
        Input("selection_B", "value"),
    )
    def _make_fig(selection_A, selection_B):
        def get_subset_embeds(selection):
            if selection is None:
                return torch.cat(paper_means), titles
            means = []
            labels = []

            for pico in picos:
                extraction_subset = defaultdict(
                    pico.default_factory,
                )

                label = []
                for k, v in pico.items():
                    if k not in selection:
                        continue
                    extraction_subset[k] = v
                    label.append("<br>".join(v))
                labels.append("<br><br>".join(label))
                means.append(model.embed_query(extraction_subset))
            return torch.cat(means), labels

        means_A, labels_A = get_subset_embeds(selection_A)
        means_B, labels_B = get_subset_embeds(selection_B)

        return plot_means_dash(means_A, means_B, labels_A, labels_B)

    app.run(debug=True, use_reloader=False)
