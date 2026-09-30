"""Generate topology and percolation figures from reproducible result files."""
import csv
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch

BASE = os.path.dirname(__file__)
RESULTS = os.path.join(BASE, "../results")
FIGS = os.path.join(BASE, "../figures")
GRAPH_PKL = os.path.join(BASE, "../data/processed/graph.pkl")
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9})


def load_csv(name):
    with open(os.path.join(RESULTS, name), newline="") as stream:
        return list(csv.DictReader(stream))


def load_graph():
    with open(GRAPH_PKL, "rb") as stream:
        return pickle.load(stream)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def numbers(rows, field):
    return np.array([float(row[field]) for row in rows])


def fig_topology():
    graph = load_graph()
    degrees = np.array([degree for _, degree in graph.degree()])
    values, counts = np.unique(degrees, return_counts=True)
    probability = counts / counts.sum()
    ccdf = np.cumsum(probability[::-1])[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    axes[0].loglog(values, probability, "o", ms=2.5, color="#1f77b4")
    axes[0].set(xlabel="Degree k", ylabel="P(k)", title="Degree distribution")
    axes[1].loglog(values, ccdf, "o", ms=2.5, color="#d62728")
    axes[1].set(xlabel="Degree k", ylabel="P(K >= k)", title="Complementary CDF")
    for axis in axes:
        axis.grid(True, which="both", ls="--", alpha=0.35)
    save(fig, "topology_degree.png")


def fig_percolation_curves():
    node = load_csv("node_percolation.csv")
    edge = load_csv("edge_percolation.csv")
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    ax.errorbar(numbers(edge, "p"), numbers(edge, "norm_mean"), yerr=numbers(edge, "norm_std"),
                fmt="s-", ms=3, lw=1, capsize=2, color="#2ca02c", label="Bond (random edge)")
    ax.errorbar(numbers(node, "p"), numbers(node, "norm_mean"), yerr=numbers(node, "norm_std"),
                fmt="o-", ms=3, lw=1, capsize=2, color="#1f77b4", label="Site (random node)")
    ax.set(xlabel="Occupation probability p", ylabel="S(p) = |LCC| / N0", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8)
    save(fig, "bond_site_percolation.png")


def fig_targeted():
    node = load_csv("node_percolation.csv")
    targeted = load_csv("targeted_removal.csv")
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    ax.errorbar(numbers(node, "p"), numbers(node, "norm_mean"), yerr=numbers(node, "norm_std"),
                fmt="o-", ms=3, lw=1, capsize=2, label="Random node removal")
    ax.plot(numbers(targeted, "p"), numbers(targeted, "norm"), "x-", ms=3, lw=1,
            color="#d62728", label="Initial-degree targeted")
    ax.axvspan(0.65, 0.70, color="#d62728", alpha=0.10, label="Targeted collapse region")
    ax.set(xlabel="Fraction of nodes remaining p", ylabel="S(p) = |LCC| / N0", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=7.5)
    save(fig, "targeted_vs_random.png")


def fig_susceptibility():
    node = load_csv("node_percolation.csv")
    edge = load_csv("edge_percolation.csv")
    targeted = load_csv("targeted_removal.csv")
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    for rows, value, label, color in [
        (edge, "chi_mean", "Bond", "#2ca02c"),
        (node, "chi_mean", "Site", "#1f77b4"),
        (targeted, "chi", "Targeted", "#d62728"),
    ]:
        p = numbers(rows, "p")
        chi = numbers(rows, value)
        ax.plot(p, chi, "-", lw=1.1, color=color, label=label)
        peak = np.argmax(chi)
        ax.plot(p[peak], chi[peak], "o", ms=4, color=color)
    ax.set(xlabel="Occupation probability p", ylabel="Finite-cluster susceptibility", xlim=(-0.02, 1.02))
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8)
    save(fig, "susceptibility.png")


def fig_components():
    node = load_csv("node_percolation.csv")
    edge = load_csv("edge_percolation.csv")
    targeted = load_csv("targeted_removal.csv")
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    ax.plot(numbers(edge, "p"), numbers(edge, "ncomp_mean"), label="Bond", color="#2ca02c")
    ax.plot(numbers(node, "p"), numbers(node, "ncomp_mean"), label="Site", color="#1f77b4")
    ax.plot(numbers(targeted, "p"), numbers(targeted, "ncomp"), label="Targeted", color="#d62728")
    ax.set(xlabel="Occupation probability p", ylabel="Number of components", xlim=(-0.02, 1.02))
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8)
    save(fig, "component_counts.png")


def fig_cluster_sizes():
    graph = load_graph()
    nodes = np.array(list(graph.nodes()))
    rng = np.random.default_rng(2026)
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    for p, color in [(0.70, "#1f77b4"), (0.40, "#ff7f0e"), (0.10, "#2ca02c")]:
        kept = rng.choice(nodes, size=round(p * len(nodes)), replace=False)
        sizes = sorted((len(component) for component in nx.connected_components(graph.subgraph(kept))), reverse=True)[1:]
        if sizes:
            bins = np.logspace(0, np.log10(max(sizes) + 1), 24)
            ax.hist(sizes, bins=bins, histtype="step", lw=1.2, color=color, label=f"Site p={p:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="Finite component size", ylabel="Count of components")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(fontsize=8)
    save(fig, "cluster_size_profiles.png")


def fig_flowchart():
    fig, ax = plt.subplots(figsize=(7.1, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    labels = [
        "Raw SNAP\nedge list",
        "Clean & validate\nundirected graph",
        "Percolation sweeps\nbond, site, targeted",
        "Components, LCC\nand susceptibility",
        "CSV tables\nand figures",
    ]
    xs = [0.2, 2.2, 4.2, 6.2, 8.2]
    for x, label in zip(xs, labels):
        box = FancyBboxPatch((x, 0.65), 1.55, 0.7, boxstyle="round,pad=0.05",
                             facecolor="#eef4fb", edgecolor="#356aa0", linewidth=1.1)
        ax.add_patch(box)
        ax.text(x + 0.775, 1.0, label, ha="center", va="center", fontsize=8)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + 1.95, 1.0), xytext=(x + 1.58, 1.0),
                    arrowprops={"arrowstyle": "->", "lw": 1, "color": "#444444"})
    save(fig, "analysis_flowchart.png")


if __name__ == "__main__":
    fig_topology()
    fig_percolation_curves()
    fig_targeted()
    fig_susceptibility()
    fig_components()
    fig_cluster_sizes()
    fig_flowchart()
