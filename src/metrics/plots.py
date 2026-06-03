import torch
import pandas as pd
import umap
import plotly.express as px


def plot_means(pico_means, paper_means):
    reducer = umap.UMAP(n_components=2)
    coords = reducer.fit_transform(torch.cat([pico_means, paper_means]).cpu().numpy())
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "labels": ["pico"] * len(pico_means) + ["paper"] * len(paper_means),
            "i": list(range(len(pico_means))) * 2,
        }
    )
    return px.scatter(df, x="x", y="y", color="labels", hover_name="i")
