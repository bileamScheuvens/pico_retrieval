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
