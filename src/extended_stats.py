#!/usr/bin/env python3
"""
Compute extended network statistics with detailed progress logging.

This version is designed to show clearly what is happening while it runs.
It writes results to:

    results/extended_stats.txt
    results/extended_stats.json

Usage:

    python -u src/extended_stats.py

If you want to skip the slowest statistics, use:

    python -u src/extended_stats.py --skip-slow
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import networkx as nx

BASE = os.path.dirname(os.path.abspath(__file__))
GRAPH_PKL = os.path.join(BASE, "..", "data", "processed", "graph.pkl")
RESULTS_DIR = os.path.join(BASE, "..", "results")


def log(message):
    """Print a timestamped message immediately."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def seconds_since(start_time):
    """Return elapsed time since start_time as a formatted string."""
    return f"{time.perf_counter() - start_time:.2f}s"


def main():
    parser = argparse.ArgumentParser(
        description="Compute extended network statistics with detailed progress output."
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip clustering coefficient, diameter, and average shortest path length.",
    )
    parser.add_argument(
        "--top-hubs",
        type=int,
        default=10,
        help="Print this many highest-degree nodes. Set to 0 to disable.",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log("=" * 70)
    log("Extended statistics computation started")
    log(f"Python version: {sys.version.split()[0]}")
    log(f"NetworkX version: {nx.__version__}")
    log(f"NumPy version: {np.__version__}")
    log("=" * 70)

    graph_path = os.path.abspath(GRAPH_PKL)
    log(f"Looking for graph file: {graph_path}")

    if not os.path.exists(graph_path):
        log("ERROR: graph.pkl not found.")
        log("Run preprocess.py first:")
        log("    python src/preprocess.py")
        sys.exit(1)

    stats = {}

    # ------------------------------------------------------------------
    # STEP 1: Load graph
    # ------------------------------------------------------------------
    log("STEP 1/9: Loading graph.pkl ...")
    step_start = time.perf_counter()

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    log(f"Graph loaded in {seconds_since(step_start)}.")

    n = G.number_of_nodes()
    m = G.number_of_edges()

    stats["nodes"] = n
    stats["edges"] = m

    log(f"Graph type: {type(G).__name__}")
    log(f"Nodes: {n}")
    log(f"Edges: {m}")

    if n == 0:
        log("ERROR: Graph has zero nodes. Cannot compute extended statistics.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 2: Degree sequence
    # ------------------------------------------------------------------
    log("STEP 2/9: Extracting degree sequence ...")
    step_start = time.perf_counter()

    degrees = np.array([d for _, d in G.degree()], dtype=np.int64)

    log(f"Degree sequence extracted in {seconds_since(step_start)}.")
    log(f"Degree sequence length: {len(degrees)}")

    if len(degrees) != n:
        log("WARNING: degree sequence length does not equal node count.")

    degree_sum = int(degrees.sum())
    stats["degree_sum"] = degree_sum

    if degree_sum != 2 * m:
        log(f"WARNING: sum(degrees)={degree_sum}, expected 2M={2 * m}.")
    else:
        log(f"Sanity check passed: sum(degrees) = {degree_sum} = 2M.")

    # ------------------------------------------------------------------
    # STEP 3: Basic degree statistics
    # ------------------------------------------------------------------
    log("STEP 3/9: Computing basic degree statistics ...")
    step_start = time.perf_counter()

    density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0

    k1 = float(degrees.mean())
    k2 = float((degrees.astype(np.float64) ** 2).mean())

    pc_configuration = None
    if k2 > k1:
        pc_configuration = k1 / (k2 - k1)

    stats["density"] = density
    stats["degree_min"] = int(degrees.min())
    stats["degree_max"] = int(degrees.max())
    stats["degree_mean"] = k1
    stats["degree_median"] = float(np.median(degrees))
    stats["degree_std_sample"] = (
        float(degrees.std(ddof=1)) if len(degrees) > 1 else 0.0
    )
    stats["degree_q1"] = float(np.percentile(degrees, 25))
    stats["degree_q3"] = float(np.percentile(degrees, 75))
    stats["degree_p90"] = float(np.percentile(degrees, 90))
    stats["degree_p95"] = float(np.percentile(degrees, 95))
    stats["degree_p99"] = float(np.percentile(degrees, 99))
    stats["second_moment_k2"] = k2
    stats["configuration_model_pc"] = pc_configuration

    if len(degrees) > 0:
        counts = np.bincount(degrees)
        stats["degree_mode"] = int(np.argmax(counts))
    else:
        stats["degree_mode"] = None

    log(f"Basic degree statistics computed in {seconds_since(step_start)}.")
    log(f"Mean degree: {k1:.6f}")
    log(f"Median degree: {stats['degree_median']}")
    log(f"Min degree: {stats['degree_min']}")
    log(f"Max degree: {stats['degree_max']}")
    log(f"Second moment <k^2>: {k2:.6f}")

    if pc_configuration is not None:
        log(
            "Configuration-model theoretical pc estimate: "
            f"{pc_configuration:.6f}"
        )
    else:
        log(
            "Could not compute configuration-model pc because "
            "<k^2> <= <k>."
        )

    # ------------------------------------------------------------------
    # STEP 4: Top hubs
    # ------------------------------------------------------------------
    if args.top_hubs > 0:
        log(f"STEP 4/9: Printing top {args.top_hubs} highest-degree nodes ...")
        step_start = time.perf_counter()

        top_hubs = sorted(
            G.degree(),
            key=lambda item: item[1],
            reverse=True,
        )[: args.top_hubs]

        for rank, (node, degree) in enumerate(top_hubs, start=1):
            log(f"  {rank:2d}. node {node}: degree {degree}")

        log(f"Top hubs printed in {seconds_since(step_start)}.")
    else:
        log("STEP 4/9: Top-hub printing disabled.")

    # ------------------------------------------------------------------
    # STEP 5: Connected components
    # ------------------------------------------------------------------
    log("STEP 5/9: Computing connected components ...")
    step_start = time.perf_counter()

    components = list(nx.connected_components(G))
    component_sizes = np.array([len(c) for c in components], dtype=np.int64)

    log(f"Connected components computed in {seconds_since(step_start)}.")
    log(f"Number of components: {len(components)}")

    if len(components) == 0:
        log("ERROR: No components found.")
        sys.exit(1)

    sorted_component_sizes = np.sort(component_sizes)[::-1]

    lcc_index = int(np.argmax(component_sizes))
    lcc_nodes = components[lcc_index]
    L = G.subgraph(lcc_nodes)

    stats["components"] = len(components)
    stats["lcc_nodes"] = int(sorted_component_sizes[0])
    stats["lcc_fraction"] = stats["lcc_nodes"] / n
    stats["second_largest_component_size"] = (
        int(sorted_component_sizes[1])
        if len(sorted_component_sizes) > 1
        else 0
    )
    stats["smallest_component_size"] = int(sorted_component_sizes[-1])
    stats["median_component_size"] = float(np.median(component_sizes))
    stats["lcc_edges"] = L.number_of_edges()
    stats["lcc_density"] = (
        (2 * stats["lcc_edges"]) /
        (stats["lcc_nodes"] * (stats["lcc_nodes"] - 1))
        if stats["lcc_nodes"] > 1
        else 0.0
    )

    log(f"Largest component size: {stats['lcc_nodes']}")
    log(f"Largest component fraction: {stats['lcc_fraction']:.6f}")
    log(f"LCC edges: {stats['lcc_edges']}")

    # ------------------------------------------------------------------
    # STEP 6: Connectivity note
    # ------------------------------------------------------------------
    log("STEP 6/9: Checking original graph connectivity ...")

    if stats["components"] == 1:
        log("The original graph is fully connected.")
    else:
        log(
            "The original graph is NOT fully connected. "
            f"It has {stats['components']} components."
        )
        log("This is why S(1) is approximately 0.9332, not 1.0.")

    # ------------------------------------------------------------------
    # STEP 7 and 8: Slow statistics
    # ------------------------------------------------------------------
    if args.skip_slow:
        log(
            "STEP 7/9: SKIPPED average clustering coefficient "
            "because --skip-slow was used."
        )
        log(
            "STEP 8/9: SKIPPED diameter and average shortest path length "
            "because --skip-slow was used."
        )

        stats["average_clustering"] = "skipped"
        stats["lcc_diameter"] = "skipped"
        stats["lcc_average_shortest_path_length"] = "skipped"

    else:
        log("STEP 7/9: Computing average clustering coefficient ...")
        log(
            "This may take 10-60 seconds and may appear frozen while running."
        )
        step_start = time.perf_counter()

        try:
            stats["average_clustering"] = float(nx.average_clustering(G))
            log(
                "Average clustering coefficient: "
                f"{stats['average_clustering']:.6f}"
            )
            log(f"Clustering computed in {seconds_since(step_start)}.")
        except Exception as e:
            stats["average_clustering"] = f"error: {e}"
            log(f"WARNING: average clustering failed: {e}")

        log(
            "STEP 8/9: Computing LCC diameter and average shortest path length ..."
        )
        log("These are BFS-based computations and may take 30-120 seconds.")

        try:
            if not nx.is_connected(L):
                log("WARNING: LCC subgraph is unexpectedly disconnected.")
                stats["lcc_diameter"] = "error: LCC disconnected"
                stats["lcc_average_shortest_path_length"] = (
                    "error: LCC disconnected"
                )
            else:
                log("  - Computing diameter ...")
                step_start = time.perf_counter()

                stats["lcc_diameter"] = int(nx.diameter(L))

                log(
                    f"  Diameter: {stats['lcc_diameter']} "
                    f"computed in {seconds_since(step_start)}."
                )

                log("  - Computing average shortest path length ...")
                step_start = time.perf_counter()

                stats["lcc_average_shortest_path_length"] = float(
                    nx.average_shortest_path_length(L)
                )

                log(
                    "  Average shortest path length: "
                    f"{stats['lcc_average_shortest_path_length']:.6f}"
                )
                log(
                    "  Average path length computed in "
                    f"{seconds_since(step_start)}."
                )

        except Exception as e:
            stats["lcc_diameter"] = f"error: {e}"
            stats["lcc_average_shortest_path_length"] = f"error: {e}"
            log(f"WARNING: LCC distance metrics failed: {e}")

    # ------------------------------------------------------------------
    # STEP 9: Save results
    # ------------------------------------------------------------------
    log("STEP 9/9: Saving results ...")

    stats["generated_at"] = datetime.now().isoformat(timespec="seconds")
    stats["runtime_seconds"] = round(time.perf_counter() - total_start, 2)

    txt_path = os.path.join(RESULTS_DIR, "extended_stats.txt")
    json_path = os.path.join(RESULTS_DIR, "extended_stats.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    log(f"Saved: {os.path.abspath(txt_path)}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    log(f"Saved: {os.path.abspath(json_path)}")

    log("=" * 70)
    log("FINAL EXTENDED STATISTICS")
    log("=" * 70)

    for key, value in stats.items():
        log(f"{key}: {value}")

    log("=" * 70)
    log(f"Total runtime: {stats['runtime_seconds']} seconds")
    log("Done.")


if __name__ == "__main__":
    main()
