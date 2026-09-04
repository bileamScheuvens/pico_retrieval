from collections import defaultdict

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.constants import EVALPATH
from src.data.scidocdata import SciDocDatamodule
from src.metrics.plots import plot_means_subsets
from src.models.artsy import ARTSY

pico_subsets = {
    "PICO": ["Patient", "Intervention", "Control", "Outcome"],
    "PIC": ["Patient", "Intervention", "Control"],
    "PIO": ["Patient", "Intervention", "Outcome"],
    "PCO": ["Patient", "Control", "Outcome"],
    "ICO": ["Intervention", "Control", "Outcome"],
}


def eval_visualisation(cfg: DictConfig):
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
    pico_means = defaultdict(list)
    for i in tqdm(range(N)):
        title, abstract = val_data[i]
        titles.append(title)
        # TODO treat prob
        paper_means.append(model.embed_paper(title, abstract))
        pico = model.extract_pico(model.join_text(title, abstract))
        for subset, categories in pico_subsets.items():
            extraction_subset = defaultdict(
                pico.default_factory, {k: v for k, v in pico.items() if k in categories}
            )
            pico_means[subset].append(model.embed_query(extraction_subset))

    pico_means = {k: torch.cat(v) for k, v in pico_means.items()}
    paper_means = torch.cat(paper_means)
    fig = plot_means_subsets(pico_means, paper_means, titles)
    fig.write_html(EVALPATH / f"eval_{cfg.index_name}.html")
