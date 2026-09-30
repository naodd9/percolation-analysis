"""Percolation analysis.
Efficient: no full graph copies for node percolation (subgraph views),
fast connected components, limited trials, vectorized RNG.
"""
import os, pickle, time, csv
import numpy as np
import networkx as nx

BASE = os.path.dirname(__file__)
GRAPH_PKL = os.path.join(BASE, "../data/processed/graph.pkl")
RESULTS = os.path.join(BASE, "../results")
P_VALS = np.round(np.arange(1.0, -0.001, -0.01), 2).tolist()
N_TRIALS = 5
SEED = 42

def component_stats(G):
    """Return component count, LCC size, and finite-cluster susceptibility."""
    sizes = sorted((len(component) for component in nx.connected_components(G)), reverse=True)
    if not sizes:
        return 0, 0, 0.0
    finite = sizes[1:]
    finite_total = sum(finite)
    susceptibility = sum(size * size for size in finite) / finite_total if finite_total else 0.0
    return len(sizes), sizes[0], susceptibility


def lcc_stats_subgraph(G, keep_nodes):
    """Compute induced-subgraph statistics without copying the base graph."""
    if len(keep_nodes) == 0:
        return 0, 0, 0, 0, 0.0
    SG = G.subgraph(keep_nodes)  # view, no copy
    m = SG.number_of_edges()
    n_comp, lcc, susceptibility = component_stats(SG)
    return len(keep_nodes), m, n_comp, lcc, susceptibility

def random_node_percolation(G, rng):
    N0 = G.number_of_nodes()
    nodes = np.array(list(G.nodes()))
    rows = []
    for p in P_VALS:
        trials = 1 if p in (0.0, 1.0) else N_TRIALS
        rec = []
        for t in range(trials):
            n_keep = int(round(p * N0))
            if n_keep >= N0:
                keep = nodes
            elif n_keep <= 0:
                keep = np.array([], dtype=nodes.dtype)
            else:
                keep = rng.choice(nodes, size=n_keep, replace=False)
            nn, mm, nc, lcc, chi = lcc_stats_subgraph(G, keep)
            rec.append((nn, mm, nc, lcc, lcc / N0, chi))
        arr = np.array(rec, dtype=float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if trials > 1 else np.zeros(6)
        rows.append((p, *mean, *std))
        print(f"[node p={p:.2f}] LCC={mean[3]:.0f} norm={mean[4]:.4f} chi={mean[5]:.2f}", flush=True)
    return rows

def random_edge_percolation(G, rng):
    N0 = G.number_of_nodes()
    M0 = G.number_of_edges()
    edges = list(G.edges())
    m_arr = np.array(edges, dtype=np.int64)
    all_nodes = list(G.nodes())
    rows = []
    for p in P_VALS:
        trials = 1 if p in (0.0, 1.0) else N_TRIALS
        rec = []
        for t in range(trials):
            m_keep = int(round(p * M0))
            if m_keep <= 0:
                # no edges: N isolated nodes
                rec.append((0, N0, 1 if N0 > 0 else 0, 1 / N0 if N0 else 0, 1.0 if N0 > 1 else 0.0))
                continue
            elif m_keep >= M0:
                n_comp, lcc, chi = component_stats(G)
                rec.append((M0, n_comp, lcc, lcc / N0, chi))
                continue
            idx = rng.choice(M0, size=m_keep, replace=False)
            H = nx.Graph()
            H.add_nodes_from(all_nodes)
            for i in idx:
                u, v = edges[i]
                H.add_edge(u, v)
            n_comp, lcc, chi = component_stats(H)
            rec.append((m_keep, n_comp, lcc, lcc / N0, chi))
        arr = np.array(rec, dtype=float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if trials > 1 else np.zeros(5)
        rows.append((p, *mean, *std))
        print(f"[edge p={p:.2f}] LCC_norm={mean[3]:.4f} chi={mean[4]:.2f}", flush=True)
    return rows

def targeted_node_removal(G):
    N0 = G.number_of_nodes()
    order = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    sorted_nodes = [n for n, d in order]
    rows = []
    for p in P_VALS:
        n_keep = int(round(p * N0))
        n_remove = N0 - n_keep
        keep = [n for n in sorted_nodes[n_remove:]] if n_keep>0 else []
        # keep is low-degree nodes
        nn, mm, nc, lcc, chi = lcc_stats_subgraph(G, keep)
        rows.append((p, nn, mm, nc, lcc, lcc / N0, chi))
        print(f"[targeted p={p:.2f}] LCC_norm={lcc/N0:.4f} chi={chi:.2f}", flush=True)
    return rows

def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def main():
    t0 = time.time()
    with open(GRAPH_PKL, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph: N={G.number_of_nodes()}, M={G.number_of_edges()}")
    rng = np.random.default_rng(SEED)
    os.makedirs(RESULTS, exist_ok=True)

    print("=== Random node percolation ===")
    nr = random_node_percolation(G, rng)
    save_csv(os.path.join(RESULTS, "node_percolation.csv"),
             ["p","nn_mean","mm_mean","ncomp_mean","lcc_mean","norm_mean","chi_mean",
              "nn_std","mm_std","ncomp_std","lcc_std","norm_std","chi_std"], nr)

    print("=== Random edge percolation ===")
    er = random_edge_percolation(G, rng)
    save_csv(os.path.join(RESULTS, "edge_percolation.csv"),
             ["p","mkeep_mean","ncomp_mean","lcc_mean","norm_mean","chi_mean",
              "mkeep_std","ncomp_std","lcc_std","norm_std","chi_std"], er)

    print("=== Targeted removal ===")
    tr = targeted_node_removal(G)
    save_csv(os.path.join(RESULTS, "targeted_removal.csv"),
             ["p","nn","mm","ncomp","lcc","norm","chi"], tr)

    dt = time.time() - t0
    with open(os.path.join(RESULTS, "runtime.txt"), 'w') as f:
        f.write(f"total_seconds: {dt:.1f}\n")
        f.write(f"trials: {N_TRIALS}, seed: {SEED}\n")
    print(f"Done in {dt:.1f}s")

if __name__ == "__main__":
    main()
