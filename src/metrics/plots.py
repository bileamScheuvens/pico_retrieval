from matplotlib.lines import Line2D
import torch
import matplotlib.pyplot as plt
import pandas as pd
import umap
import plotly.express as px
import plotly.graph_objects as go


def plot_means_plotly(pico_means, paper_means, paper_titles):
    reducer = umap.UMAP(n_components=2)
    coords = reducer.fit_transform(torch.cat([pico_means, paper_means]).cpu().numpy())
    num = len(pico_means)
    fig = go.Figure()

    def add_scatter(label, coords, symbol):
        fig.add_scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            name=label,
            marker={"color": list(range(num)), "colorscale": "Turbo"},
            showlegend=False,
            text=paper_titles,
        )

    add_scatter("pico", coords[:num], "x")
    add_scatter("paper", coords[num:], "diamond")
    return fig


def plot_means_plt(pico_means, paper_means, paper_titles):
    reducer = umap.UMAP(n_components=2)
    coords = reducer.fit_transform(torch.cat([pico_means, paper_means]).cpu().numpy())
    n = len(pico_means)
    fig, ax = plt.subplots()
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": ["pico"] * n + ["paper"] * n,
            "i": list(range(n)) * 2,
            "title": paper_titles * 2,
        }
    )

    for label, group in df.groupby("label"):
        ax.scatter(
            group["x"],
            group["y"],
            c=group["i"],
            cmap="turbo",
            marker="D" if label == "paper" else "x",
        )
    # positioning is hard
    # ax.legend(
    #     [Line2D([0], [0])] * n,
    #     [f"{i}: {paper_titles[i][:75]}" for i in range(n)],
    #     loc="upper center",
    #     bbox_to_anchor=(0, 1),
    # )

    return fig
