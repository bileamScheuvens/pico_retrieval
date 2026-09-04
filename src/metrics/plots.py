import torch
from sklearn.preprocessing import MinMaxScaler
import plotly.colors as pc
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
    query_means: dict, paper_means: torch.Tensor, paper_titles, seed=161
) -> go.Figure:
    reducer = umap.UMAP(n_components=2, random_state=seed).fit(
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
            marker={
                "color": list(range(num)),
                "colorscale": "Turbo",
                "symbol": symbol,
                "line": {
                    "width": 1,
                    "color": list(range(num)),
                    "colorscale": "Turbo",
                },
            },
            text=[f"[{label}]: {title}" for title in paper_titles],
        )

    add_scatter("paper", paper_means, "diamond")

    for (subset, means), symbol in zip(
        query_means.items(), ["asterisk", "line-ns", "line-ew", "line-ne", "line-nw"]
    ):
        add_scatter(subset, means, symbol)
    return fig


def plot_means_dash(
    means_A: torch.Tensor, means_B: torch.Tensor, labels_A, labels_B, seed=161
) -> go.Figure:
    num = len(labels_A)
    fig = go.Figure()

    def add_scatter(labels, coords, selector, symbol):
        coords = umap.UMAP(n_components=2, random_state=seed).fit_transform(
            coords.cpu().numpy()
        )
        coords = MinMaxScaler().fit_transform(coords)
        fig.add_scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            name=selector,
            marker={
                "color": list(range(num)),
                "colorscale": "Turbo",
                "symbol": symbol,
                "line": {
                    "width": 1,
                    "color": list(range(num)),
                    "colorscale": "Turbo",
                },
            },
            text=labels,
        )
        return coords

    coords_A = add_scatter(labels_A, means_A, "A", "diamond")
    coords_B = add_scatter(labels_B, means_B, "B", "asterisk")

    arrows = []
    arrow_colors = pc.sample_colorscale("Turbo", num)
    for i in range(num):
        arrows.append(
            {
                "x": coords_A[i, 0],
                "y": coords_A[i, 1],
                "ax": coords_B[i, 0],
                "ay": coords_B[i, 1],
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "arrowcolor": arrow_colors[i],
                "showarrow": True,
                "opacity": 0.1,
            }
        )
    fig.update_layout(annotations=arrows)

    return fig
