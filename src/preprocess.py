"""Preprocess ca-HepPh: clean, validate, and compute basic statistics."""
import gzip, os, pickle
from collections import Counter
import networkx as nx

RAW = os.path.join(os.path.dirname(__file__), "../data/raw/ca-HepPh.txt.gz")
PROCESSED_EDGELIST = os.path.join(os.path.dirname(__file__), "../data/processed/ca-HepPh_edgelist.txt")
GRAPH_PKL = os.path.join(os.path.dirname(__file__), "../data/processed/graph.pkl")
STATS_TXT = os.path.join(os.path.dirname(__file__), "../data/processed/stats.txt")

def load_raw():
    total = 0
    selfloops = 0
    with gzip.open(RAW, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            total += 1
            u, v = map(int, line.strip().split())
            if u == v:
                selfloops += 1
    return total, selfloops

def build_graph():
    G = nx.Graph()
    seen = set()
    selfloops_removed = 0
    with gzip.open(RAW, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            u, v = map(int, line.strip().split())
            if u == v:
                selfloops_removed += 1
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                continue  # duplicate undirected edge (covers both directions)
            seen.add((a, b))
            G.add_edge(a, b)
    # The downloaded adjacency list contains both orientations of every non-loop
    # pair.  Canonicalizing (u, v) above therefore produces one undirected edge.
    # SNAP reports 12,008 node IDs, while two isolated IDs do not occur in an
    # edge row; this graph intentionally represents the observed edge list.
    return G, selfloops_removed

def main():
    os.makedirs(os.path.dirname(PROCESSED_EDGELIST), exist_ok=True)
    total, _ = load_raw()
    G, selfloops = build_graph()
    n = G.number_of_nodes()
    m = G.number_of_edges()
    comps = list(nx.connected_components(G))
    n_comp = len(comps)
    lcc = max(comps, key=len) if comps else set()
    lcc_size = len(lcc)
    avg_deg = 2 * m / n if n else 0
    degs = [d for _, d in G.degree()]
    deg_counter = Counter(degs)
    max_deg = max(degs) if degs else 0

    print(f"Raw lines: {total}, self-loops removed: {selfloops}")
    print(f"Nodes: {n} (>10000? {n > 10000})")
    print(f"Edges (undirected): {m} (>20000? {m > 20000})")
    print(f"Components: {n_comp}, LCC: {lcc_size} ({lcc_size/n:.4f}), avg_deg: {avg_deg:.2f}, max_deg: {max_deg}")
    assert n > 10000 and m > 20000, "Dataset does not meet size requirements!"

    # save edgelist
    with open(PROCESSED_EDGELIST, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    with open(GRAPH_PKL, 'wb') as f:
        pickle.dump(G, f)
    with open(STATS_TXT, 'w') as f:
        f.write(f"nodes: {n}\nedges: {m}\ncomponents: {n_comp}\nlcc: {lcc_size}\n")
        f.write(f"lcc_frac: {lcc_size/n:.6f}\navg_degree: {avg_deg:.4f}\nmax_degree: {max_deg}\n")
        f.write(f"selfloops_removed: {selfloops}\nraw_lines: {total}\n")
        f.write(f"raw_nonloop_lines: {total - selfloops}\n")
        f.write(f"unique_undirected_nonloop_edges: {m}\n")
        f.write(f"orientation_or_duplicate_lines_collapsed: {total - selfloops - m}\n")
    # Save the degree distribution, creating its output directory first so this
    # script also works from a fresh checkout.
    import csv
    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "degree_distribution.csv"), 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(["degree", "count"])
        for d in sorted(deg_counter):
            w.writerow([d, deg_counter[d]])
    print(f"Saved {PROCESSED_EDGELIST}, {GRAPH_PKL}")

if __name__ == "__main__":
    main()
