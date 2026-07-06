import torch
import umap
import plotly.graph_objects as go


def plot_means(pico_means, paper_means, paper_titles) -> go.Figure:
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
            text=[f"[{label}]: {title}" for title in paper_titles],
        )

    add_scatter("pico", coords[:num], "x")
    add_scatter("paper", coords[num:], "diamond")
    return fig


def plot_means_subsets(
    query_means: dict, paper_means: torch.Tensor, paper_titles
) -> go.Figure:
    reducer = umap.UMAP(n_components=2).fit(
        torch.cat([query_means["PICO"], paper_means]).cpu().numpy()
    )
    num = len(paper_titles)
    fig = go.Figure()

    def add_scatter(label, coords, symbol):
        coords = reducer.transform(coords)
        fig.add_scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            name=label,
            marker={"color": list(range(num)), "colorscale": "Turbo", "symbol": symbol},
            showlegend=False,
            text=[f"[{label}]: {title}" for title in paper_titles],
        )

    add_scatter("paper", paper_means, "diamond")
    for subset, means in query_means.items():
        add_scatter(subset, means, "x")
    return fig
