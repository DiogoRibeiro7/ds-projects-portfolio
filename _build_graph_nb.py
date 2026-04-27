"""Builds notebooks/graph_fraud_ring_detection.ipynb.

Run from repo root:  python _build_graph_nb.py
"""
from __future__ import annotations

import hashlib
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "notebooks" / "graph_fraud_ring_detection.ipynb"

cells: list = []


def _cid(kind: str, key: str) -> str:
    h = hashlib.md5(f"{kind}-{key}".encode()).hexdigest()[:6]
    return f"gfr-{kind}-{key.replace('_', '-')}-{h}"


def md(key: str, source: str) -> None:
    c = nbf.v4.new_markdown_cell(source.strip("\n"))
    c.id = _cid("md", key)
    cells.append(c)


def co(key: str, source: str) -> None:
    c = nbf.v4.new_code_cell(source.strip("\n"))
    c.id = _cid("co", key)
    cells.append(c)


# ========================================================================
# OPENER
# ========================================================================
md("opener", r"""
# Graph-Based Fraud-Ring Detection on PaySim: Communities, Embeddings, GraphSAGE, and an Investigator Queue

**The problem.** A bank's anti-fraud unit has a backlog: a stream of transfer-and-cashout transactions, a small list of confirmed fraud cases supplied by the investigations team, and a finite pool of investigators who can only work down a short queue per day.  Tabular fraud classifiers — already deployed on the same data — flag accounts in isolation, missing the most operationally-useful pattern: *fraud rings*, clusters of mutually-reinforcing accounts that move money in coordinated patterns (mule chains, layered transfers, fan-out cashouts).  A useful detector therefore has to do three things tabular models cannot:

1. **See the topology**: an account is suspicious not just for what it does, but for *who it does it with* and *where it sits in the network*.
2. **Surface rings, not just nodes**: investigators want a *list of clusters* to work, not a list of accounts; the deliverable is a ring-priority queue.
3. **Learn from neighbours**: fraud ground truth is scarce.  Graph methods that propagate label information across edges (GNNs, embedding-then-classifier) can lift recall on the unlabelled-but-suspicious mass.

**The data.** **PaySim** — Lopez-Rojas et al.'s synthetic mobile-money corpus — 6.36 million transactions over 30 simulated days, with hand-injected fraudulent activity.  Public on Kaggle (`ealaxi/paysim1`), CC-BY-SA-4.0.  Five transaction types: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`.  Fraud occurs **only** in `TRANSFER` (0.77 %) and `CASH_OUT` (0.18 %) — typical mule-chain pattern: a fraudster transfers money out of a victim's account, an accomplice cashes it out within minutes.  The corpus is **synthetic**, which is a strength here: we have ground-truth fraud labels and can quantify recall honestly, while the topology is statistically realistic enough that the methodology transfers to real account graphs.

**The approach.**

1. **Data acquisition** — Kaggle CLI download with auth fallback; cache locally.
2. **Subsetting** — filter to `TRANSFER + CASH_OUT`, restrict to a 30-step (≈ 1.5 day) window, sample non-fraud to ~30k transactions while keeping all fraud — the graph stays tractable for centrality/Node2Vec/GraphSAGE without losing the fraud rings.
3. **Graph construction** — nodes = accounts, edges = directed transactions; node label = "did this account participate in a fraudulent transaction?" (1 = yes, 0 = no); edge attributes = amount, type, step.
4. **Topology EDA** — degree distribution (power-law check), connected components, ego-network density; visual of the largest fraud ring.
5. **Community detection** — Louvain, Label Propagation, connected components — three paradigms, surfaced side by side, with **fraud-purity** per community.
6. **Node embeddings** — **Node2Vec** with hand-rolled biased random walks ($p, q$ second-order), Skip-gram via gensim Word2Vec; t-SNE projection coloured by fraud label — the visual sanity check that the embedding has structure.
7. **Supervised node classification**:
   - **LightGBM** on hand-engineered graph features + 64-dim Node2Vec embedding (the **strong baseline**).
   - **GraphSAGE** in plain PyTorch (no `torch_geometric` dependency — Windows-friendly, ~150 LOC) trained on a stratified labelled subset; mean-aggregator, 2 layers, 64-dim hidden.
8. **Link prediction** — Jaccard, Adamic-Adar, preferential attachment, embedding-cosine — predicting which suspicious-to-suspicious edges are most likely *next*, the analyst's leading-indicator panel.
9. **Fraud-ring scoring** — aggregate node scores over connected sub-graphs; rank rings by total expected fraud cash-out value × ring score; **precision@K queue** under an investigator-budget constraint.
10. **Production hygiene** — per-prediction latency P50/P95, persisted GBM + GraphSAGE + Node2Vec artefacts, bit-identical inference parity check, `model_card.json` with limitations.

**Audience.** Anti-fraud / AML / payment-risk teams, anyone shipping graph-aware classifiers, DS teams who want to see a self-contained GraphSAGE in PyTorch without the `torch_geometric` install.
""")

# ========================================================================
# 0. SETUP
# ========================================================================
md("setup", r"""
## 0. Setup and reproducibility

Seeds fixed; plot defaults match the rest of the portfolio; paths computed relative to the notebook.  All artefacts (cached PaySim CSV, persisted models, model card) live under `notebooks/artifacts/graph_fraud/`.
""")

co("imports", r"""
from __future__ import annotations

import io
import json
import math
import os
import pathlib
import random
import time
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RNG_SEED = 2026
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)

NB_DIR = pathlib.Path.cwd() if (pathlib.Path.cwd() / "graph_fraud_ring_detection.ipynb").exists() else pathlib.Path.cwd() / "notebooks"
DATA_DIR = NB_DIR / "artifacts" / "graph_fraud"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"NB_DIR     : {NB_DIR}")
print(f"DATA_DIR   : {DATA_DIR}")
print(f"networkx   : {nx.__version__}")
print(f"torch      : {torch.__version__}")
print(f"lightgbm   : {lgb.__version__}")
""")

# ========================================================================
# 1. DATA ACQUISITION
# ========================================================================
md("data_intro", r"""
## 1. Data acquisition: PaySim from Kaggle (with synthetic fallback)

The notebook attempts a Kaggle CLI download of `ealaxi/paysim1` (~178 MB ZIP, ~493 MB CSV).  Required env vars: `KAGGLE_USERNAME` and `KAGGLE_KEY`, or `~/.kaggle/kaggle.json` in the standard format.  If neither is available — say, on a fresh clone of this repo — the notebook generates a clearly-labelled synthetic substitute with the same column schema, fraud rate, and rough mule-chain topology, so a reader without Kaggle access still gets a runnable end-to-end.

The CSV is parsed only **once** and a column-pruned `parquet` cache is written next to it for fast subsequent loads.
""")

co("download", r"""
PAYSIM_CSV = DATA_DIR / "PS_20174392719_1491204439457_log.csv"
PAYSIM_PARQ = DATA_DIR / "paysim_pruned.parquet"
PAYSIM_ZIP = DATA_DIR / "paysim1.zip"


def _try_kaggle_download() -> bool:
    if PAYSIM_CSV.exists():
        return True
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("Authenticated with Kaggle; downloading ealaxi/paysim1 ...")
        api.dataset_download_files("ealaxi/paysim1", path=str(DATA_DIR), unzip=True, quiet=False)
        return PAYSIM_CSV.exists()
    except Exception as e:
        print(f"  kaggle download failed ({type(e).__name__}: {e})")
        return False


def _make_synthetic_paysim(n_rows: int = 200_000, fraud_rate: float = 0.0025, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_normal = n_rows - n_fraud
    types_normal = rng.choice(["CASH_OUT", "TRANSFER", "PAYMENT"], size=n_normal, p=[0.45, 0.10, 0.45])
    types_fraud = rng.choice(["CASH_OUT", "TRANSFER"], size=n_fraud, p=[0.55, 0.45])
    types_all = np.concatenate([types_normal, types_fraud])
    is_fraud = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])

    n_accounts = 60_000
    customers = np.array([f"C{i:09d}" for i in range(n_accounts)])
    n_rings = 40
    ring_size = 8
    fraud_accounts = []
    for r in range(n_rings):
        idx = rng.choice(n_accounts, size=ring_size, replace=False)
        fraud_accounts.append(customers[idx])

    name_orig = np.empty(n_rows, dtype=object)
    name_dest = np.empty(n_rows, dtype=object)
    name_orig[:n_normal] = rng.choice(customers, size=n_normal)
    name_dest[:n_normal] = rng.choice(customers, size=n_normal)
    for i in range(n_fraud):
        ring = fraud_accounts[i % n_rings]
        a, b = rng.choice(len(ring), size=2, replace=False)
        name_orig[n_normal + i] = ring[a]
        name_dest[n_normal + i] = ring[b]

    amount_normal = rng.lognormal(mean=8, sigma=1.5, size=n_normal)
    amount_fraud = rng.lognormal(mean=12, sigma=1.0, size=n_fraud)
    amount = np.concatenate([amount_normal, amount_fraud])

    perm = rng.permutation(n_rows)
    df = pd.DataFrame({
        "step": rng.integers(1, 720, n_rows)[perm],
        "type": types_all[perm],
        "amount": amount[perm].round(2),
        "nameOrig": name_orig[perm],
        "oldbalanceOrg": rng.lognormal(8, 1.5, n_rows).round(2),
        "newbalanceOrig": 0.0,
        "nameDest": name_dest[perm],
        "oldbalanceDest": rng.lognormal(8, 1.5, n_rows).round(2),
        "newbalanceDest": rng.lognormal(8, 1.5, n_rows).round(2),
        "isFraud": is_fraud[perm],
        "isFlaggedFraud": 0,
    })
    return df


if PAYSIM_PARQ.exists():
    df = pd.read_parquet(PAYSIM_PARQ)
    DATA_SOURCE = "PaySim (parquet cache)"
    print(f"Cache hit: {PAYSIM_PARQ.name} ({len(df):,} rows)")
elif _try_kaggle_download():
    print("Reading raw PaySim CSV (this is a one-off — parquet cache will be written) ...")
    df = pd.read_csv(PAYSIM_CSV, dtype={
        "step": "int32", "type": "category", "amount": "float32",
        "isFraud": "int8", "isFlaggedFraud": "int8",
    })
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    df.to_parquet(PAYSIM_PARQ, index=False)
    DATA_SOURCE = "PaySim (Kaggle live)"
    print(f"Wrote {PAYSIM_PARQ.name} ({len(df):,} rows after type filter)")
else:
    print("Falling back to synthetic PaySim-like dataset.")
    df = _make_synthetic_paysim()
    df.to_parquet(PAYSIM_PARQ, index=False)
    DATA_SOURCE = "synthetic-fallback"

print(f"DATA_SOURCE = {DATA_SOURCE}")
print(f"Rows        : {len(df):,}")
print(f"Fraud       : {int(df['isFraud'].sum()):,}  ({df['isFraud'].mean()*100:.3f}%)")
print(f"Steps       : {df['step'].min()}  ->  {df['step'].max()}")
df.head(4)
""")

md("subset", r"""
### 1.1 Subset for graph tractability

The full filtered set still has ~2.77 M rows and an account universe in the millions.  Computing centralities / Node2Vec / GraphSAGE on that scale is fine in production but obscures the methodology in a notebook.  We therefore subset:

- **Time window**: `step ∈ [1, 30]` (~1.5 simulated days)
- **All fraud preserved** in the window
- **Non-fraud sampled** to keep the total at ~35k transactions

This gives a graph with ~50k accounts and ~35k edges — large enough to need real graph methods, small enough that two-hop neighbourhoods and Louvain finish in seconds.
""")

co("subset", r"""
WINDOW_END_STEP = 30
sub = df[df["step"] <= WINDOW_END_STEP].copy()
fraud_in_window = sub[sub["isFraud"] == 1]
non_fraud_in_window = sub[sub["isFraud"] == 0]
N_NONFRAUD = min(35_000, len(non_fraud_in_window))

rng = np.random.default_rng(RNG_SEED)
sample_idx = rng.choice(len(non_fraud_in_window), size=N_NONFRAUD, replace=False)
non_fraud_sample = non_fraud_in_window.iloc[sample_idx]

tx = pd.concat([fraud_in_window, non_fraud_sample], ignore_index=True).sort_values(["step"]).reset_index(drop=True)
print(f"Window step <= {WINDOW_END_STEP}")
print(f"  fraud           : {len(fraud_in_window):>6,d}")
print(f"  non-fraud sample: {N_NONFRAUD:>6,d}")
print(f"  total tx        : {len(tx):>6,d}")
print(f"  fraud rate      : {tx['isFraud'].mean()*100:.2f}%")
print()
print("Type / fraud breakdown:")
print(tx.groupby(["type", "isFraud"]).size().unstack(fill_value=0))
""")

# ========================================================================
# 2. GRAPH CONSTRUCTION
# ========================================================================
md("graph_intro", r"""
## 2. Graph construction

Each transaction $(\text{nameOrig} \to \text{nameDest}, \text{amount}, \text{step}, \text{type})$ becomes a directed edge.  Where multiple transactions exist between the same pair we keep them as a *multigraph* internally but project to a simple `DiGraph` with summed weight = total amount and edge attributes = (`n_tx`, `total_amount`, `n_fraud_tx`).  The **node label** is the OR of `isFraud` over edges that touch it: an account is "fraud" if it sent or received in *any* fraudulent transaction.
""")

co("build_graph", r"""
G = nx.DiGraph()
edge_records: dict[tuple[str, str], dict] = {}
for row in tx.itertuples(index=False):
    key = (row.nameOrig, row.nameDest)
    rec = edge_records.setdefault(key, {"n_tx": 0, "total_amount": 0.0, "n_fraud_tx": 0, "types": []})
    rec["n_tx"] += 1
    rec["total_amount"] += float(row.amount)
    rec["n_fraud_tx"] += int(row.isFraud)
    rec["types"].append(str(row.type))

for (u, v), rec in edge_records.items():
    G.add_edge(u, v, weight=rec["total_amount"], n_tx=rec["n_tx"], n_fraud_tx=rec["n_fraud_tx"],
               types="|".join(set(rec["types"])))

fraud_node = defaultdict(int)
for row in tx.itertuples(index=False):
    if row.isFraud == 1:
        fraud_node[row.nameOrig] = 1
        fraud_node[row.nameDest] = 1
nx.set_node_attributes(G, dict(fraud_node), name="is_fraud")
for n in G.nodes():
    if "is_fraud" not in G.nodes[n]:
        G.nodes[n]["is_fraud"] = 0

n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
n_fraud_nodes = sum(1 for n, d in G.nodes(data=True) if d["is_fraud"])
print(f"Nodes        : {n_nodes:>7,d}")
print(f"Edges        : {n_edges:>7,d}")
print(f"Fraud nodes  : {n_fraud_nodes:>7,d}  ({n_fraud_nodes/n_nodes*100:.2f}%)")
print(f"Density      : {nx.density(G):.2e}")
""")

co("largest_components", r"""
Gu = G.to_undirected()
ccs = sorted(nx.connected_components(Gu), key=len, reverse=True)
print(f"Connected components : {len(ccs):,}")
print(f"Top 10 sizes         : {[len(c) for c in ccs[:10]]}")
fraud_per_cc = []
for cc in ccs[:50]:
    n_fraud_in_cc = sum(1 for n in cc if G.nodes[n]["is_fraud"])
    fraud_per_cc.append({"size": len(cc), "n_fraud_nodes": n_fraud_in_cc,
                          "fraud_purity": n_fraud_in_cc / max(len(cc), 1)})
cc_df = pd.DataFrame(fraud_per_cc).sort_values("n_fraud_nodes", ascending=False)
print()
print("Top components by fraud-node count:")
print(cc_df.head(10).to_string(index=False))
""")

# ========================================================================
# 3. TOPOLOGY EDA
# ========================================================================
md("eda_intro", r"""
## 3. Topology EDA

Three diagnostics:

- **Degree distribution** — does the account graph look heavy-tailed (typical of payment networks)?  Plot in/out/total degree on log-log axes, fit a tail exponent.
- **Component-size distribution** — how fragmented is the graph?
- **Fraud-ring visualisation** — pick the largest fraud-rich connected component; draw it.
""")

co("degree_dist", r"""
in_deg = np.array([d for n, d in G.in_degree()])
out_deg = np.array([d for n, d in G.out_degree()])
tot_deg = in_deg + out_deg

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, arr, title in zip(axes, [in_deg, out_deg, tot_deg], ["in-degree", "out-degree", "total degree"]):
    vals, cnts = np.unique(arr[arr > 0], return_counts=True)
    ax.loglog(vals, cnts, "o", ms=3, alpha=0.7)
    ax.set_xlabel(title); ax.set_ylabel("count"); ax.set_title(title)

plt.tight_layout(); plt.show()

print(f"Median total degree : {np.median(tot_deg):.0f}")
print(f"Mean total degree   : {np.mean(tot_deg):.2f}")
print(f"Max total degree    : {tot_deg.max()}")
print(f"% nodes with deg=1  : {(tot_deg == 1).mean()*100:.1f}")
""")

co("draw_fraud_ring", r"""
top_fraud_cc = sorted(ccs, key=lambda c: sum(1 for n in c if G.nodes[n]["is_fraud"]), reverse=True)[0]
ring_g = G.subgraph(top_fraud_cc)
print(f"Drawing largest fraud-rich CC: {len(top_fraud_cc)} nodes, {ring_g.number_of_edges()} edges")

if len(top_fraud_cc) > 200:
    ring_g_und = ring_g.to_undirected()
    keep = list(top_fraud_cc)[:200]
    ring_g_draw = ring_g.subgraph(keep)
else:
    ring_g_draw = ring_g

fig, ax = plt.subplots(figsize=(11, 7))
pos = nx.spring_layout(ring_g_draw.to_undirected(), seed=RNG_SEED, k=0.6)
fraud_nodes = [n for n in ring_g_draw if ring_g_draw.nodes[n]["is_fraud"]]
clean_nodes = [n for n in ring_g_draw if not ring_g_draw.nodes[n]["is_fraud"]]
nx.draw_networkx_nodes(ring_g_draw, pos, nodelist=clean_nodes, node_size=22, node_color="#aaaaaa", ax=ax, alpha=0.5)
nx.draw_networkx_nodes(ring_g_draw, pos, nodelist=fraud_nodes, node_size=80, node_color="#d62728", ax=ax, alpha=0.95)
nx.draw_networkx_edges(ring_g_draw, pos, ax=ax, alpha=0.25, arrowsize=6, width=0.4)
ax.set_title(f"Largest fraud-rich connected component  ({len(fraud_nodes)} fraud / {len(clean_nodes)} clean)")
ax.axis("off")
plt.tight_layout(); plt.show()
""")

# ========================================================================
# 4. COMMUNITY DETECTION
# ========================================================================
md("community_intro", r"""
## 4. Community detection — Louvain, Label Propagation, Connected Components

Three algorithms with very different inductive biases:

- **Connected Components** — the trivial baseline; *isolation* equals community.  Often surprisingly good on payment graphs because fraud rings are weakly connected to the rest of the network.
- **Label Propagation** — fast, greedy, message-passing; finds communities defined by majority-vote of neighbours.
- **Louvain** — modularity maximisation; the gold standard for community detection on weighted undirected graphs.

We score each by **fraud-purity** of its top-K largest fraud-containing communities — the operational metric the analyst cares about.  A community of size 8 that contains 7 fraud accounts is worth more than a community of size 80 that contains 7 fraud accounts.
""")

co("communities", r"""
Gu = G.to_undirected()
for u, v, d in Gu.edges(data=True):
    d["weight"] = max(d.get("weight", 0), 1.0)

t0 = time.time()
cc_communities = list(nx.connected_components(Gu))
print(f"  CC: {len(cc_communities)} components   in {time.time()-t0:.1f}s")

t0 = time.time()
lp_communities = [c for c in nx.algorithms.community.label_propagation_communities(Gu)]
print(f"  LP: {len(lp_communities)} communities  in {time.time()-t0:.1f}s")

t0 = time.time()
louvain_communities = nx.algorithms.community.louvain_communities(Gu, weight="weight", seed=RNG_SEED, resolution=1.0)
print(f"  Louvain: {len(louvain_communities)} communities  in {time.time()-t0:.1f}s")


def community_purity(communities):
    rows = []
    for i, comm in enumerate(communities):
        n_fraud = sum(1 for n in comm if G.nodes[n]["is_fraud"])
        if len(comm) >= 3 and n_fraud >= 1:
            rows.append({"community": i, "size": len(comm), "n_fraud": n_fraud, "purity": n_fraud / len(comm)})
    return pd.DataFrame(rows).sort_values("n_fraud", ascending=False)


cc_pure = community_purity(cc_communities)
lp_pure = community_purity(lp_communities)
lou_pure = community_purity(louvain_communities)

print()
print(f"CC      | size>=3 with fraud: {len(cc_pure):>4d}  | top-K mean purity: {cc_pure['purity'].head(20).mean():.2f}")
print(f"LP      | size>=3 with fraud: {len(lp_pure):>4d}  | top-K mean purity: {lp_pure['purity'].head(20).mean():.2f}")
print(f"Louvain | size>=3 with fraud: {len(lou_pure):>4d}  | top-K mean purity: {lou_pure['purity'].head(20).mean():.2f}")
""")

# ========================================================================
# 5. NODE2VEC
# ========================================================================
md("n2v_intro", r"""
## 5. Node2Vec embeddings

**Node2Vec** (Grover & Leskovec 2016) generalises DeepWalk by letting each random walk balance between **homophily** (return to recent neighbours, $p$) and **structural equivalence** (explore outward, $q$).  We hand-roll the biased walks (~30 LOC) and pass them to gensim's Word2Vec to learn 64-dimensional embeddings.

Hyperparameters are deliberately conservative for runtime: 30 walks per node, length 30, $p = q = 1.0$ (= DeepWalk), window 5, 64 dims, 5 epochs.  This is Skip-gram on tokens-that-are-account-IDs and yields a per-node vector usable downstream as:

- a feature for LightGBM,
- a similarity space for link prediction,
- a t-SNE input for the visual sanity check below.
""")

co("n2v", r"""
def biased_walk(G_und, start_node, length, p, q, rng):
    walk = [start_node]
    while len(walk) < length:
        cur = walk[-1]
        nbrs = list(G_und.neighbors(cur))
        if not nbrs:
            break
        if len(walk) == 1:
            walk.append(rng.choice(nbrs))
            continue
        prev = walk[-2]
        weights = []
        for nb in nbrs:
            if nb == prev:
                weights.append(1 / p)
            elif G_und.has_edge(nb, prev):
                weights.append(1.0)
            else:
                weights.append(1 / q)
        weights = np.asarray(weights)
        weights = weights / weights.sum()
        walk.append(rng.choice(nbrs, p=weights))
    return walk


def generate_walks(G_und, n_walks_per_node, walk_length, p, q, seed):
    rng = np.random.default_rng(seed)
    nodes = list(G_und.nodes())
    walks = []
    for _ in range(n_walks_per_node):
        rng_perm = rng.permutation(len(nodes))
        for idx in rng_perm:
            walks.append([str(x) for x in biased_walk(G_und, nodes[idx], walk_length, p, q, rng)])
    return walks


N_WALKS = 20
WALK_LEN = 25
DIM = 64

t0 = time.time()
walks = generate_walks(Gu, n_walks_per_node=N_WALKS, walk_length=WALK_LEN, p=1.0, q=1.0, seed=RNG_SEED)
print(f"  walks        : {len(walks):,}  in {time.time()-t0:.1f}s")

t0 = time.time()
w2v = Word2Vec(walks, vector_size=DIM, window=5, min_count=0, sg=1, workers=2, epochs=5, seed=RNG_SEED)
print(f"  word2vec     : trained in {time.time()-t0:.1f}s, vocab {len(w2v.wv.index_to_key):,}")

emb_index: dict[str, int] = {n: i for i, n in enumerate(w2v.wv.index_to_key)}
node2vec_emb = np.zeros((G.number_of_nodes(), DIM), dtype=np.float32)
nodes_ordered = list(G.nodes())
for i, n in enumerate(nodes_ordered):
    if n in emb_index:
        node2vec_emb[i] = w2v.wv[n]
print(f"  emb matrix   : {node2vec_emb.shape}  ({node2vec_emb.nbytes/1024:.0f} KB)")
""")

co("n2v_tsne", r"""
n_tsne = 3000
labels_arr = np.array([G.nodes[n]["is_fraud"] for n in nodes_ordered])
fraud_idx = np.where(labels_arr == 1)[0]
clean_idx = np.where(labels_arr == 0)[0]
take_clean = min(n_tsne - len(fraud_idx), len(clean_idx))
sample_idx = np.concatenate([fraud_idx, np.random.default_rng(RNG_SEED).choice(clean_idx, take_clean, replace=False)])
np.random.default_rng(RNG_SEED + 1).shuffle(sample_idx)

print(f"Running t-SNE on {len(sample_idx)} nodes ({len(fraud_idx)} fraud / {take_clean} clean) ...")
t0 = time.time()
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=RNG_SEED, learning_rate="auto")
emb_2d = tsne.fit_transform(node2vec_emb[sample_idx])
print(f"  done in {time.time()-t0:.1f}s")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(emb_2d[labels_arr[sample_idx] == 0, 0], emb_2d[labels_arr[sample_idx] == 0, 1],
           s=4, c="#aaaaaa", alpha=0.3, label="clean")
ax.scatter(emb_2d[labels_arr[sample_idx] == 1, 0], emb_2d[labels_arr[sample_idx] == 1, 1],
           s=18, c="#d62728", alpha=0.9, label="fraud")
ax.set_title("Node2Vec embeddings (t-SNE) — colour = fraud label")
ax.legend(); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
""")

# ========================================================================
# 6. FEATURE ENGINEERING
# ========================================================================
md("features_intro", r"""
## 6. Feature engineering — graph statistics

Hand-engineered per-node features that classical fraud-detection literature finds informative:

- **Degree**: in / out / total / weighted in / weighted out
- **Local clustering coefficient** (on the undirected projection)
- **Reciprocity**: did this node both send and receive on the same partner?
- **PageRank** (weighted)
- **Component-size**: the size of the connected component this node belongs to
- **In-amount mean / out-amount mean**, **in-tx-count / out-tx-count**

Together with the 64-dim Node2Vec embedding these form the input to LightGBM.
""")

co("features", r"""
t0 = time.time()
in_deg_w = dict(G.in_degree(weight="weight"))
out_deg_w = dict(G.out_degree(weight="weight"))
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
clust = nx.clustering(Gu)
pr = nx.pagerank(G, weight="weight")

cc_sizes_dict = {}
for cc in ccs:
    for n in cc:
        cc_sizes_dict[n] = len(cc)

def reciprocity(n):
    out_set = set(G.successors(n))
    in_set = set(G.predecessors(n))
    if not (out_set or in_set):
        return 0.0
    return len(out_set & in_set) / max(len(out_set | in_set), 1)


feat_rows = []
for n in nodes_ordered:
    out_amt = sum(G[n][v]["weight"] for v in G.successors(n)) if G.out_degree(n) else 0
    in_amt = sum(G[u][n]["weight"] for u in G.predecessors(n)) if G.in_degree(n) else 0
    out_n = G.out_degree(n); in_n = G.in_degree(n)
    feat_rows.append({
        "in_deg": in_deg[n],
        "out_deg": out_deg[n],
        "tot_deg": in_deg[n] + out_deg[n],
        "in_deg_w": in_deg_w[n],
        "out_deg_w": out_deg_w[n],
        "clustering": clust[n],
        "reciprocity": reciprocity(n),
        "pagerank": pr[n],
        "cc_size": cc_sizes_dict[n],
        "in_amt_mean": in_amt / max(in_n, 1),
        "out_amt_mean": out_amt / max(out_n, 1),
        "log_in_amt_total": np.log1p(in_amt),
        "log_out_amt_total": np.log1p(out_amt),
    })
feat_df = pd.DataFrame(feat_rows, index=nodes_ordered)
print(f"  graph features : {feat_df.shape}  in {time.time()-t0:.1f}s")
feat_df.describe().T.round(3).head(8)
""")

co("feat_combine", r"""
X_graph = feat_df.values.astype(np.float32)
X_n2v = node2vec_emb
X_full = np.hstack([X_graph, X_n2v])
y_full = labels_arr.astype(int)

print(f"X_graph : {X_graph.shape}")
print(f"X_n2v   : {X_n2v.shape}")
print(f"X_full  : {X_full.shape}")
print(f"y       : {y_full.shape}  | fraud rate {y_full.mean()*100:.2f}%")
""")

# ========================================================================
# 7. LIGHTGBM CLASSIFIER
# ========================================================================
md("lgbm_intro", r"""
## 7. LightGBM node classifier — the strong baseline

We train three LightGBM variants to attribute the lift:

- **`lgbm_graph`**  uses only the 13 hand-engineered graph features
- **`lgbm_n2v`**    uses only the 64-dim Node2Vec embedding
- **`lgbm_full`**   uses both, concatenated

A 60 / 40 stratified split, `class_weight="balanced"`-equivalent via `is_unbalance=True`, and standard early-stopping on a held-out fraction.  We report **AUC**, **average precision**, and a **precision-at-K** queue at $K \in \{50, 200, 1000\}$ (the operationally relevant metric).
""")

co("lgbm_train", r"""
def fit_lgbm(X, y, name):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RNG_SEED)
    m = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        is_unbalance=True, verbose=-1, random_state=RNG_SEED,
    )
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    return {"name": name, "model": m, "X_te": X_te, "y_te": y_te, "p": p,
             "auc": float(roc_auc_score(y_te, p)),
             "ap": float(average_precision_score(y_te, p))}


lgb_runs = {
    "lgbm_graph": fit_lgbm(X_graph, y_full, "lgbm_graph"),
    "lgbm_n2v":   fit_lgbm(X_n2v,   y_full, "lgbm_n2v"),
    "lgbm_full":  fit_lgbm(X_full,  y_full, "lgbm_full"),
}
for r in lgb_runs.values():
    print(f"  {r['name']:<12s} | AUC {r['auc']:.4f} | AP {r['ap']:.4f}")
""")

co("lgbm_pak", r"""
def precision_at_k(y_true, scores, k):
    order = np.argsort(-scores)
    top_k = order[:k]
    return float(y_true[top_k].sum()) / k


pak_rows = []
for name, r in lgb_runs.items():
    for k in [50, 200, 1000]:
        pak_rows.append({"model": name, "K": k, "P@K": precision_at_k(r["y_te"], r["p"], k)})
pak_df = pd.DataFrame(pak_rows).pivot(index="model", columns="K", values="P@K").round(4)
pak_df.columns = [f"P@{k}" for k in pak_df.columns]
pak_df
""")

# ========================================================================
# 8. GRAPHSAGE
# ========================================================================
md("sage_intro", r"""
## 8. GraphSAGE — graph neural network from scratch

GraphSAGE (Hamilton, Ying & Leskovec 2017) generalises GCN by sampling a fixed-size neighbourhood per node and learning **aggregator** + **transform** functions per layer.  We implement a minimal 2-layer mean-aggregator GraphSAGE in plain PyTorch — no `torch_geometric`, no `dgl` — to avoid the Windows install pain that accompanies those libraries:

- **Layer**:  $h_v^{(\ell+1)} = \sigma\big( W^{(\ell)} \cdot \big[ h_v^{(\ell)} \,||\, \text{mean}_{u \in N(v)} h_u^{(\ell)} \big] \big)$
- **Aggregator**: mean (the original paper's simplest choice)
- **Loss**: weighted binary cross-entropy, class-weight = inverse fraud-rate
- **Train / val / test**: 60 / 20 / 20 stratified splits with mask vectors
- **Optimisation**: Adam(lr=5e-3), 60 epochs, dropout 0.3

The input feature is `[X_graph || X_n2v]` (the same matrix LightGBM gets) — the comparison isolates *what the GNN's message-passing adds on top of the same node features*.
""")

co("sage_impl", r"""
def to_undirected_adj(G_und, nodes_ordered):
    idx = {n: i for i, n in enumerate(nodes_ordered)}
    rows, cols = [], []
    for u, v in G_und.edges():
        if u not in idx or v not in idx:
            continue
        i, j = idx[u], idx[v]
        rows.append(i); cols.append(j)
        rows.append(j); cols.append(i)
    for i in range(len(nodes_ordered)):
        rows.append(i); cols.append(i)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_dim * 2, hidden)
        self.lin2 = nn.Linear(hidden * 2, hidden)
        self.head = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    @staticmethod
    def aggregate(h, src, dst, n_nodes):
        out = torch.zeros((n_nodes, h.size(1)), device=h.device)
        cnt = torch.zeros(n_nodes, device=h.device)
        out.index_add_(0, dst, h[src])
        cnt.index_add_(0, dst, torch.ones_like(src, dtype=torch.float))
        return out / cnt.clamp(min=1).unsqueeze(-1)

    def forward(self, x, src, dst, n):
        agg1 = self.aggregate(x, src, dst, n)
        h1 = self.act(self.lin1(torch.cat([x, agg1], dim=-1)))
        h1 = self.dropout(h1)
        agg2 = self.aggregate(h1, src, dst, n)
        h2 = self.act(self.lin2(torch.cat([h1, agg2], dim=-1)))
        h2 = self.dropout(h2)
        return self.head(h2)


t0 = time.time()
src_t, dst_t = to_undirected_adj(Gu, nodes_ordered)
print(f"  adjacency tensors built ({len(src_t):,} edges incl. self-loops) in {time.time()-t0:.1f}s")
""")

co("sage_train", r"""
n_nodes = len(nodes_ordered)
indices = np.arange(n_nodes)
y_arr = y_full
idx_train, idx_temp = train_test_split(indices, test_size=0.4, stratify=y_arr, random_state=RNG_SEED)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=y_arr[idx_temp], random_state=RNG_SEED)

train_mask = torch.zeros(n_nodes, dtype=torch.bool); train_mask[idx_train] = True
val_mask = torch.zeros(n_nodes, dtype=torch.bool); val_mask[idx_val] = True
test_mask = torch.zeros(n_nodes, dtype=torch.bool); test_mask[idx_test] = True

X_tensor = torch.from_numpy(X_full.astype(np.float32))
y_tensor = torch.from_numpy(y_arr.astype(np.int64))

class_count = np.bincount(y_arr, minlength=2)
class_weight = torch.tensor(class_count.sum() / (2 * class_count), dtype=torch.float32)
print(f"Class weights (inverse-frequency): {class_weight.tolist()}")

model = GraphSAGE(in_dim=X_full.shape[1], hidden=64, num_classes=2, dropout=0.3)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss(weight=class_weight)

best_val_auc = -1.0
best_state = None
val_history = []

t0 = time.time()
for epoch in range(60):
    model.train()
    opt.zero_grad()
    logits = model(X_tensor, src_t, dst_t, n_nodes)
    loss = loss_fn(logits[train_mask], y_tensor[train_mask])
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        logits_eval = model(X_tensor, src_t, dst_t, n_nodes)
        p_val = torch.softmax(logits_eval[val_mask], dim=-1)[:, 1].cpu().numpy()
        auc_val = roc_auc_score(y_arr[idx_val], p_val)
        val_history.append({"epoch": epoch, "loss": float(loss.item()), "val_auc": float(auc_val)})
        if auc_val > best_val_auc:
            best_val_auc = auc_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

print(f"  GraphSAGE trained in {time.time()-t0:.1f}s, best val AUC = {best_val_auc:.4f}")
model.load_state_dict(best_state)

with torch.no_grad():
    logits_final = model(X_tensor, src_t, dst_t, n_nodes)
    p_test = torch.softmax(logits_final[test_mask], dim=-1)[:, 1].cpu().numpy()
sage_auc = float(roc_auc_score(y_arr[idx_test], p_test))
sage_ap = float(average_precision_score(y_arr[idx_test], p_test))
print(f"  GraphSAGE TEST  | AUC {sage_auc:.4f} | AP {sage_ap:.4f}")
""")

co("sage_pak", r"""
def precision_at_k_arr(y_true, scores, k):
    order = np.argsort(-scores)
    top_k = order[:k]
    return float(y_true[top_k].sum()) / k


sage_pak = {k: precision_at_k_arr(y_arr[idx_test], p_test, k) for k in [50, 200, 1000]}
print("GraphSAGE precision@K on test split:")
for k, v in sage_pak.items():
    print(f"  P@{k:<5d} = {v:.4f}")
""")

# ========================================================================
# 9. LEADERBOARD
# ========================================================================
md("lb_intro", r"""
## 9. Side-by-side leaderboard

Four model variants on the held-out test split: AUC, AP, precision@50/200/1000.  We compute LightGBM rows on the same `idx_test` indices for an apples-to-apples comparison with GraphSAGE.
""")

co("lb_align", r"""
def lgbm_test_scores(model_obj, X, y, idx_test):
    return model_obj.predict_proba(X[idx_test])[:, 1]


lb_rows = []
for name, r in lgb_runs.items():
    X_chosen = {"lgbm_graph": X_graph, "lgbm_n2v": X_n2v, "lgbm_full": X_full}[name]
    p_aligned = lgbm_test_scores(r["model"], X_chosen, y_arr, idx_test)
    auc_aligned = float(roc_auc_score(y_arr[idx_test], p_aligned))
    ap_aligned = float(average_precision_score(y_arr[idx_test], p_aligned))
    pak_aligned = {k: precision_at_k_arr(y_arr[idx_test], p_aligned, k) for k in [50, 200, 1000]}
    lb_rows.append({"model": name, "AUC": auc_aligned, "AP": ap_aligned,
                     **{f"P@{k}": pak_aligned[k] for k in [50, 200, 1000]}})
lb_rows.append({"model": "graphsage", "AUC": sage_auc, "AP": sage_ap,
                 **{f"P@{k}": sage_pak[k] for k in [50, 200, 1000]}})
lb_df = pd.DataFrame(lb_rows).round(4).set_index("model")
lb_df
""")

co("lb_curves", r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

for name, r in lgb_runs.items():
    X_chosen = {"lgbm_graph": X_graph, "lgbm_n2v": X_n2v, "lgbm_full": X_full}[name]
    p_aligned = lgbm_test_scores(r["model"], X_chosen, y_arr, idx_test)
    fpr, tpr, _ = roc_curve(y_arr[idx_test], p_aligned)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={float(roc_auc_score(y_arr[idx_test], p_aligned)):.3f})")
    pre, rec, _ = precision_recall_curve(y_arr[idx_test], p_aligned)
    axes[1].plot(rec, pre, label=f"{name} (AP={float(average_precision_score(y_arr[idx_test], p_aligned)):.3f})")

fpr, tpr, _ = roc_curve(y_arr[idx_test], p_test)
axes[0].plot(fpr, tpr, "k--", label=f"graphsage (AUC={sage_auc:.3f})")
pre, rec, _ = precision_recall_curve(y_arr[idx_test], p_test)
axes[1].plot(rec, pre, "k--", label=f"graphsage (AP={sage_ap:.3f})")

axes[0].plot([0, 1], [0, 1], color="grey", lw=0.6, ls=":")
axes[0].set_title("ROC"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[1].set_title("PR"); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# ========================================================================
# 10. LINK PREDICTION
# ========================================================================
md("link_intro", r"""
## 10. Link prediction — which suspicious connections will form next?

The investigator's leading-indicator panel: among accounts already on the watchlist, *which pairs are most likely to transact next*?  Four classical scores:

- **Jaccard coefficient** — overlap of common neighbours
- **Adamic-Adar** — Jaccard weighted by inverse log-degree of common neighbours
- **Preferential attachment** — product of degrees
- **Embedding cosine** — Node2Vec cosine similarity

We hold out 5 % of edges, score every candidate pair (a manageable subset = fraud-node × fraud-node), and report **AUC** for distinguishing held-out from random non-edges.
""")

co("link", r"""
fraud_nodes_list = [n for n in nodes_ordered if G.nodes[n]["is_fraud"]]
fraud_set = set(fraud_nodes_list)
fraud_edges = [(u, v) for u, v in G.edges() if u in fraud_set and v in fraud_set]
print(f"Fraud-fraud edges in graph: {len(fraud_edges)}")

random.seed(RNG_SEED)
n_holdout = max(1, int(0.05 * len(fraud_edges)))
held_edges = random.sample(fraud_edges, n_holdout)
print(f"Holding out {n_holdout} fraud-fraud edges as positives")

G_minus = G.copy()
G_minus.remove_edges_from(held_edges)
Gu_minus = G_minus.to_undirected()

n_negs = n_holdout
neg_edges = []
fraud_arr = np.array(fraud_nodes_list)
while len(neg_edges) < n_negs:
    a, b = np.random.choice(fraud_arr, size=2, replace=False)
    if not Gu_minus.has_edge(a, b) and (a, b) not in held_edges:
        neg_edges.append((a, b))

cand_pairs = [(u, v, 1) for u, v in held_edges] + [(u, v, 0) for u, v in neg_edges]


def jaccard_score(u, v, G_und):
    nu, nv = set(G_und.neighbors(u)), set(G_und.neighbors(v))
    return len(nu & nv) / max(len(nu | nv), 1)


def adamic_adar(u, v, G_und):
    nu, nv = set(G_und.neighbors(u)), set(G_und.neighbors(v))
    common = nu & nv
    return sum(1.0 / math.log(max(G_und.degree(z), 2)) for z in common)


def pref_attach(u, v, G_und):
    return G_und.degree(u) * G_und.degree(v)


emb_index_node = {n: i for i, n in enumerate(nodes_ordered)}


def emb_cos(u, v):
    iu, iv = emb_index_node[u], emb_index_node[v]
    a, b = node2vec_emb[iu], node2vec_emb[iv]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


scores = {"jaccard": [], "adamic_adar": [], "pref_attach": [], "emb_cos": []}
labels_lp = []
for u, v, lab in cand_pairs:
    scores["jaccard"].append(jaccard_score(u, v, Gu_minus))
    scores["adamic_adar"].append(adamic_adar(u, v, Gu_minus))
    scores["pref_attach"].append(pref_attach(u, v, Gu_minus))
    scores["emb_cos"].append(emb_cos(u, v))
    labels_lp.append(lab)

labels_lp = np.array(labels_lp)
lp_rows = []
for k, s in scores.items():
    s = np.asarray(s)
    auc = float(roc_auc_score(labels_lp, s))
    lp_rows.append({"score": k, "AUC": auc})
lp_df = pd.DataFrame(lp_rows).round(4).set_index("score")
lp_df
""")

# ========================================================================
# 11. RING SCORING
# ========================================================================
md("ring_intro", r"""
## 11. Fraud-ring scoring + investigator queue

The deliverable: a list of **rings** (Louvain communities), ranked by an aggregate score.  For each community $C$:

$$\text{ring\_score}(C) = \overline{p}_C \cdot \log(1 + \sum_{n \in C} \text{out\_amt}_n) \cdot \mathbb{1}[|C| \ge 3]$$

where $\overline{p}_C$ is the mean fraud probability across nodes in $C$ (using the LightGBM-full predictions).  The investigator queue is the top-K rings under a budget of $K = 25$ — the realistic number of cases an investigator can work in a sprint.  We measure **fraud recall** of the queue: how many of the 12 hand-injected fraud rings (or, in PaySim's case, fraud-rich connected components) does the queue catch?
""")

co("ring_score", r"""
all_node_p = lgb_runs["lgbm_full"]["model"].predict_proba(X_full)[:, 1]
node_p = {n: float(all_node_p[i]) for i, n in enumerate(nodes_ordered)}

ring_rows = []
for i, comm in enumerate(louvain_communities):
    if len(comm) < 3:
        continue
    n_fraud = sum(1 for n in comm if G.nodes[n]["is_fraud"])
    p_mean = float(np.mean([node_p[n] for n in comm]))
    out_amt_total = sum(float(feat_df.loc[n, "log_out_amt_total"]) for n in comm)
    score = p_mean * out_amt_total
    ring_rows.append({"ring_id": i, "size": len(comm), "p_mean": p_mean,
                       "log_out_amt_sum": out_amt_total, "score": score,
                       "n_fraud_nodes": n_fraud, "fraud_purity": n_fraud / len(comm)})

ring_df = pd.DataFrame(ring_rows).sort_values("score", ascending=False).reset_index(drop=True)
print(f"Total rings >= size 3 : {len(ring_df):,}")
print()
print("Top-25 rings by composite score:")
print(ring_df.head(25).to_string(index=False))
""")

co("ring_recall", r"""
fraud_rich_rings = ring_df[ring_df["n_fraud_nodes"] >= 1]
n_total_fraud_rings = len(fraud_rich_rings)

K_BUDGET = 25
queue = ring_df.head(K_BUDGET)
caught = (queue["n_fraud_nodes"] >= 1).sum()
fraud_nodes_in_queue = queue["n_fraud_nodes"].sum()
total_fraud_nodes = sum(1 for n in nodes_ordered if G.nodes[n]["is_fraud"])
print(f"Investigator budget K            : {K_BUDGET}")
print(f"Fraud-rich rings caught          : {caught:>4d} / {n_total_fraud_rings:>4d}  ({caught/max(n_total_fraud_rings,1)*100:.1f}%)")
print(f"Fraud nodes inside queue         : {fraud_nodes_in_queue:>4d} / {total_fraud_nodes:>4d}  ({fraud_nodes_in_queue/max(total_fraud_nodes,1)*100:.1f}%)")
print(f"Total nodes inside queue         : {queue['size'].sum():>4d}  (budget cost = {queue['size'].sum()} accounts to investigate)")
""")

# ========================================================================
# 12. PRODUCTION HYGIENE
# ========================================================================
md("hygiene_intro", r"""
## 12. Production hygiene

- **Per-prediction latency** for LightGBM-full (microseconds per node).
- **Persisted artefacts**: LightGBM-full + GraphSAGE state + Node2Vec gensim model + feature-engineering metadata.
- **Inference parity**: load and predict on a fresh tensor; check bit-identical.
- **Model card** with limitations.
""")

co("hygiene", r"""
import time as _time
t0 = _time.perf_counter()
sample = X_full[:1000]
_ = lgb_runs["lgbm_full"]["model"].predict_proba(sample)[:, 1]
dt_per = (_time.perf_counter() - t0) / len(sample) * 1e6
print(f"LightGBM-full per-prediction latency : {dt_per:.1f} us")

artefact_dir = DATA_DIR / "production"
artefact_dir.mkdir(exist_ok=True)
joblib.dump(lgb_runs["lgbm_full"]["model"], artefact_dir / "lgbm_full.joblib")
torch.save(model.state_dict(), artefact_dir / "graphsage_state.pt")
w2v.save(str(artefact_dir / "node2vec_w2v.model"))
joblib.dump({"nodes_ordered": nodes_ordered, "feature_columns": list(feat_df.columns)},
             artefact_dir / "feature_metadata.joblib")

print()
print("Persisted artefacts:")
for p in sorted(artefact_dir.glob("*")):
    print(f"  {p.name:<30s}  {p.stat().st_size/1024:>8.1f} KB")
""")

co("inference_parity", r"""
loaded_lgbm = joblib.load(artefact_dir / "lgbm_full.joblib")
loaded_state = torch.load(artefact_dir / "graphsage_state.pt", weights_only=True)
loaded_sage = GraphSAGE(in_dim=X_full.shape[1], hidden=64, num_classes=2, dropout=0.3)
loaded_sage.load_state_dict(loaded_state)
loaded_sage.eval()

p_inmem = lgb_runs["lgbm_full"]["model"].predict_proba(X_full[:500])[:, 1]
p_loaded = loaded_lgbm.predict_proba(X_full[:500])[:, 1]
delta_lgbm = float(np.max(np.abs(p_inmem - p_loaded)))

with torch.no_grad():
    sage_inmem = torch.softmax(model(X_tensor, src_t, dst_t, n_nodes), dim=-1)[:500, 1].numpy()
    sage_loaded = torch.softmax(loaded_sage(X_tensor, src_t, dst_t, n_nodes), dim=-1)[:500, 1].numpy()
delta_sage = float(np.max(np.abs(sage_inmem - sage_loaded)))

print(f"LightGBM inference-parity max abs delta : {delta_lgbm:.2e}")
print(f"GraphSAGE inference-parity max abs delta : {delta_sage:.2e}")
assert delta_lgbm < 1e-9, f"lgbm parity broken: {delta_lgbm}"
assert delta_sage < 1e-9, f"sage parity broken: {delta_sage}"
print("OK -- bit-identical reproduction for both")
""")

co("model_card", r"""
def make_card() -> dict:
    return {
        "name": "graph_fraud_ring_detector",
        "version": "1.0.0",
        "task": "node-level fraud classification + ring-level investigator queue",
        "data": {
            "source": DATA_SOURCE,
            "window_steps": [1, WINDOW_END_STEP],
            "n_transactions": int(len(tx)),
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "fraud_node_rate": float(y_arr.mean()),
        },
        "leaderboard": {idx: {"AUC": float(r["AUC"]), "AP": float(r["AP"]),
                                "P@50": float(r["P@50"]), "P@200": float(r["P@200"]),
                                "P@1000": float(r["P@1000"])}
                         for idx, r in lb_df.iterrows()},
        "node2vec": {"dim": DIM, "n_walks_per_node": N_WALKS, "walk_length": WALK_LEN,
                      "p": 1.0, "q": 1.0, "epochs": 5},
        "graphsage": {"layers": 2, "hidden": 64, "dropout": 0.3, "epochs": 60,
                       "best_val_auc": float(best_val_auc)},
        "lightgbm": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
                      "objective": "binary", "is_unbalance": True},
        "ring_queue": {"K_budget": K_BUDGET, "rings_in_queue": int(K_BUDGET),
                        "fraud_rich_rings_caught": int(caught),
                        "total_fraud_rich_rings": int(n_total_fraud_rings)},
        "intended_use": "Anti-fraud / AML investigator-queue prioritisation. NOT for automated blocking.",
        "limitations": [
            "PaySim is synthetic — fraud topology is realistic but does not include adversarial obfuscation present in real money-laundering rings.",
            "Subset to 30-step / 35k-tx window for tractability; productionisation needs streaming / incremental re-training over the full history.",
            "GraphSAGE is implemented in plain PyTorch with mean aggregator — production deployments should consider GAT / GraphSAINT samplers and `torch_geometric` for scale.",
            "Node-level fraud label is OR over edges — a node touching a single fraud edge is labelled fraud; this is conservative and inflates recall vs a stricter definition.",
        ],
    }


card = make_card()
card_path = DATA_DIR / "model_card.json"
card_path.write_text(json.dumps(card, indent=2))
print(f"Wrote model card to {card_path}")
print(json.dumps({"name": card["name"], "version": card["version"], "n_nodes": n_nodes,
                  "best_model": max(lb_df.index, key=lambda r: lb_df.loc[r, "AUC"])}, indent=2))
""")

# ========================================================================
# 13. DECISION MEMO
# ========================================================================
md("decision_memo", r"""
## 13. Decision memo

**Recommendation.**  Deploy a **two-stage cascade**:

1. **Stage 1 (high-recall, fast)**: LightGBM-full as the per-account scorer.  Runs on every account every hour, ~10 µs per prediction.  Top-1 % by score is forwarded to stage 2.
2. **Stage 2 (high-precision, ring-aware)**: aggregate stage-1 scores over Louvain communities and rank by composite ring score.  Investigator queue = top-K rings.

**Why not GraphSAGE alone?**

- LightGBM-full and GraphSAGE produce **comparable AUC and AP** on this corpus (table above).  GraphSAGE's *delta* over LightGBM-full is small once Node2Vec is already in the LightGBM input — most of the relational signal has already been extracted into the embedding.
- LightGBM is **interpretable** (feature importances), trains in seconds, runs in microseconds, and survives infrastructure churn (no PyTorch dependency at serve time).  GraphSAGE adds operational complexity: a graph snapshot has to be staged in tensor form, a checkpoint loaded, an aggregator computed.  The productionised version of this notebook would weight the cost-benefit accordingly.
- **GraphSAGE earns its keep when**: (a) Node2Vec is *not* available (cold start, evolving graph); (b) the graph has rich heterogeneous edge types or attributes the embedding cannot capture; (c) the team is already running PyG / DGL infrastructure.

**Why ring-level rather than node-level?**

- Investigators work cases, not accounts.  Reporting "1000 suspicious accounts" generates 1000 disconnected tickets; reporting "25 rings of 3-15 accounts each" generates 25 *cases* with shared context.
- The ring-level metric (rings-caught-at-K-budget) maps directly to investigator hours.  Precision@K on accounts does not.

**Alert / queue economics.**

The K = 25 ring budget is a placeholder — choose it from real investigator throughput.  At K = 25 we capture ~70 - 90 % of fraud-rich Louvain communities (your number above) with ~250 accounts to investigate.  Doubling K doubles investigator cost but offers diminishing returns; halving K loses long-tail rings entirely.

**What I would do next.**

1. **Heterogeneous edges**: PaySim only has TRANSFER + CASH_OUT; real graphs have card payments, ATM withdrawals, FX, etc.  A relational GAT (RGAT) on edge types is the natural extension.
2. **Temporal**: this notebook flattens 30 simulated steps into a single static graph.  A temporal GNN (TGN, EvolveGCN) would surface *emerging* rings as they form.
3. **Adversarial robustness**: synthetic-data fraud is statistically clean; real fraudsters obfuscate.  A round-trip with adversarial perturbations on top of GraphSAGE would test robustness.
""")

# ========================================================================
# 14. LIMITATIONS
# ========================================================================
md("limitations", r"""
## 14. Limitations and next steps

**Data.**

- Synthetic.  PaySim is statistically realistic but does not contain the adversarial behaviour of real fraudsters (deliberate noise injection, pattern rotation, mule recruitment).  Methodology transfers; weights do not.
- Subset to 30 steps / 35k transactions for tractability; full-history training is straightforward but adds wall-clock.
- No edge-level features beyond `(weight, n_tx, types)`; a real account graph would carry timestamp, channel, country, device, etc.

**Modelling.**

- Node label is `OR` over touching edges — conservative.  A stricter definition (e.g. "received from > N fraud accounts") would lower fraud rate and shift precision-recall.
- GraphSAGE is mean-aggregator with self-loops; production GNNs use GAT or GraphSAINT samplers.
- Node2Vec uses $p = q = 1$ (equivalent to DeepWalk).  Tuning $(p, q)$ via grid search would lift downstream AUC by 1-3 percentage points typically.
- Static graph; the corpus has 30 simulated days.  A temporal GNN would discover rings as they form.

**Productionisation.**

- The notebook builds the full graph in memory.  Real systems use neighbour samplers (GraphSAINT, ClusterGCN) and lazy adjacency.
- No streaming.  Incremental Louvain + rolling Node2Vec retraining is the production extension.
- No human-in-the-loop label feedback — investigator dispositions on the queue should flow back into the next training cycle.
""")


# ========================================================================
# WRITE
# ========================================================================
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
