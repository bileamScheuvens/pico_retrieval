import json
from src.constants import EXPORTPATH
from collections import defaultdict
from functools import lru_cache

import torch
from dash import Dash, Input, Output, callback, dcc
from omegaconf import DictConfig
from tqdm import tqdm
from random import Random

from src.data.scidocdata import SciDocDatamodule, SciDocDataset
from src.metrics.plots import plot_means_dash
from src.models.artsy import ARTSY


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
    dataset = SciDocDataset(cfg.data)

    indices = list(range(len(dataset)))
    Random(cfg.data.seed).shuffle(indices)

    N = cfg.N
    pmids = []
    titles = []
    paper_means = []
    picos = []

    # precompute paper embed and pico extraction
    for i in tqdm(indices[:N]):
        title, abstract = dataset[i]
        pmids.append(dataset.pmids[i])
        titles.append(title)
        # TODO treat prob
        paper_means.append(model.embed_paper(title, abstract))
        picos.append(model.extract_pico(model.join_text(title, abstract)))

    with (EXPORTPATH / "labels" / f"{N}_pmids.json").open("w") as f:
        json.dump({"pmids": pmids}, f)

    @lru_cache
    def _cached_subset_embed(selection):
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

    def _get_subset_embeds(selection):
        if selection is None:
            return torch.cat(paper_means), titles
        return _cached_subset_embed(tuple(selection))

    @callback(
        Output("fig", "figure"),
        Input("selection_A", "value"),
        Input("selection_B", "value"),
    )
    def _make_fig(selection_A, selection_B):

        means_A, labels_A = _get_subset_embeds(selection_A)
        means_B, labels_B = _get_subset_embeds(selection_B)

        return plot_means_dash(
            means_A,
            means_B,
            labels_A,
            labels_B,
            selection_A,
            selection_B,
            cfg=cfg,
            min_dist=0.05,
            n_neighbors=50,
            metric="cosine",
        )

    app.run(debug=True, use_reloader=False, threaded=False)
