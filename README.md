# Percolation Analysis on the ca-HepPh Citation Network

Reproducible study of **site (random node), bond (random edge), and degree-targeted percolation** on the SNAP *ca-HepPh* collaboration network, tracking the largest connected component (LCC) and finite-cluster susceptibility to locate the percolation threshold.

## Installation

```bash
git clone https://github.com/naodd9/percolation-analysis.git
cd percolation-analysis

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+. Dependencies: `networkx`, `numpy`, `matplotlib`, `pandas`, `scipy`.

## Quick Start

Run the full pipeline in order:

```bash
python src/download_data.py     # 1. fetch ca-HepPh.txt.gz -> data/raw/
python src/preprocess.py        # 2. clean + validate -> data/processed/graph.pkl
python src/percolation.py       # 3. bond/site/targeted sweeps -> results/*.csv
python src/extended_stats.py    # 4. extended statistics (optional)
python src/visualize_network.py # 5. network snapshot -> figures/
python src/analysis.py          # 6. all analysis figures -> figures/
```

Steps 3–6 depend on `graph.pkl`; step 6 also reads the CSVs from step 3. A full run takes roughly 2–3 minutes (sweeps ≈ 85 s); add `--skip-slow` to step 4 to bypass clustering/diameter.

## Usage

**Extended statistics** — optional flags:

```bash
python -u src/extended_stats.py --skip-slow --top-hubs 5
```

- `--skip-slow` — skip clustering coefficient, diameter, average shortest path
- `--top-hubs N` — print top-N highest-degree nodes (`0` disables)

**Custom sweeps** — edit the module-level constants in `src/percolation.py`:

```python
P_VALS = np.round(np.arange(1.0, -0.001, -0.01), 2).tolist()  # p grid
N_TRIALS = 5      # Monte-Carlo trials per p
SEED = 42         # reproducibility
```

**Using the graph directly:**

```python
import pickle
from src.percolation import component_stats

with open("data/processed/graph.pkl", "rb") as f:
    G = pickle.load(f)

n_comp, lcc, chi = component_stats(G)
print(f"components={n_comp}, LCC={lcc}, chi={chi:.2f}")
```

## Project Structure

```
src/download_data.py      # SNAP download
src/preprocess.py         # clean, validate, build graph.pkl, degree distribution
src/percolation.py        # site / bond / targeted sweeps + susceptibility
src/extended_stats.py     # extended topology statistics (txt + json)
src/visualize_network.py  # sampled LCC spring-layout render
src/analysis.py           # percolation, susceptibility, topology figures
data/raw/                 # downloaded edge list
data/processed/           # graph.pkl, edgelist, stats.txt
results/                  # CSV tables, extended_stats.{txt,json}, runtime.txt
figures/                  # generated PNGs (300 dpi)
```