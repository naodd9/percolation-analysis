"""Generate a readable sampled visualization of the largest connected component."""

import os
import pickle

import numpy as np
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
GRAPH_PKL = os.path.join(BASE, "..", "data", "processed", "graph.pkl")
FIGS = os.path.join(BASE, "..", "figures")


def main():
    os.makedirs(FIGS, exist_ok=True)

    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)

    lcc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc_nodes)

    rng = np.random.default_rng(42)

    sample_size = min(2000, H.number_of_nodes())
    nodes = np.array(list(H.nodes()))

    sampled_nodes = rng.choice(nodes, size=sample_size, replace=False)

    # Force inclusion of the highest-degree hubs so the visualization
    # preserves the hub structure.
    top_hubs = [
        node
        for node, degree in sorted(
            H.degree(),
            key=lambda item: item[1],
            reverse=True
        )[:100]
    ]

    keep_nodes = np.unique(np.concatenate([sampled_nodes, top_hubs]))
    S = H.subgraph(keep_nodes)

    degree = dict(S.degree())

    pos = nx.spring_layout(
        S,
        k=0.18,
        iterations=30,
        seed=42,
    )

    node_sizes = [
        min(300, 10 + 2 * degree[node])
        for node in S.nodes()
    ]

    node_colors = [degree[node] for node in S.nodes()]

    plt.figure(figsize=(7, 5), dpi=300)

    nx.draw_networkx_edges(
        S,
        pos,
        alpha=0.05,
        width=0.35,
        edge_color="#999999",
    )

    nodes = nx.draw_networkx_nodes(
        S,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="viridis",
    )

    plt.axis("off")
    plt.colorbar(nodes, label="Degree", shrink=0.7)
    plt.title(
        f"Sampled LCC visualization "
        f"({S.number_of_nodes()} nodes, {S.number_of_edges()} edges)"
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGS, "network_visualization.png"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
