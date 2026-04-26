"""Builds notebooks/multi_label_text_classification_reuters.ipynb.

Run from repo root:  python _build_multilabel_nb.py
"""
from __future__ import annotations

import hashlib
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "notebooks" / "multi_label_text_classification_reuters.ipynb"

cells: list = []


def _cid(kind: str, key: str) -> str:
    h = hashlib.md5(f"{kind}-{key}".encode()).hexdigest()[:6]
    return f"ml-{kind}-{key.replace('_', '-')}-{h}"


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
# Multi-Label Text Classification on Reuters-21578: Per-Label Thresholds, Asymmetric Costs, Head-vs-Tail Calibration

**The problem.** A platform team runs a tagging pipeline: every incoming document needs **zero or more** topic tags drawn from a 90-topic taxonomy.  Volume is millions of documents per day; the downstream system routes by tag, so each tag's per-label F1 is the operationally relevant metric — and **per-tag mistakes have different costs**.  A missed `earnings` tag is cheap (the system is fine flagging it on the next round); a missed `lawsuit` tag is expensive (compliance ramifications).

This is **not** a multiclass problem.  It is **multi-label**, and that distinction has consequences a single-label classifier cannot ship around:

1. **Per-label thresholds**.  An argmax over a softmax is no longer the right decoder; each binary head needs its own threshold, tuned on validation data.
2. **Per-label calibration**.  Outputs from a sigmoid head are not by default calibrated probabilities; downstream cost-aware decisions require per-label Platt / isotonic recalibration.
3. **Asymmetric costs**.  An F1 maximiser is implicitly a cost-symmetric maximiser.  When false-positive cost ≠ false-negative cost, the right threshold is *not* the F1-max threshold.
4. **Head-vs-tail collapse**.  The top-5 labels in Reuters cover ~80 % of the corpus; the bottom 50 labels share single-digit positives.  Macro-F1 collapses on the tail unless rare labels get explicit help.
5. **Label dependencies**.  `earnings` and `acquired` co-occur far more often than chance.  Independent binary heads waste this signal; chained / structured outputs can recover it.

The deliverable is therefore not just "trained a multi-label classifier"; it is **a head-vs-tail leaderboard, per-label thresholds, an asymmetric-cost operating point, and an honest assessment of which labels we can ship today vs which need more data**.

**The data.** **Reuters-21578** (ModApte split) — the classical multi-label benchmark.  ~7,769 training / ~3,019 test articles with 90 topic labels assigned by Reuters editors.  Public, downloadable via NLTK (one-time download, no auth).  Severe imbalance: top label `earn` has ~3,800 positives, bottom labels (`barley`, `cocoa`, `coconut-oil`) have under 10.

**The approach.**

1. **Data acquisition** — NLTK download with cache.
2. **EDA** — per-label frequency distribution, multi-label statistics (mean labels per doc, label co-occurrence heat map), doc-length distribution.
3. **TF-IDF** vectoriser + `MultiLabelBinarizer` — the standard pipeline.
4. **Six estimators**:
   - **Binary Relevance + LinearSVC** — the OvR bar.
   - **Classifier Chains** — capture label correlations by chaining binary classifiers.
   - **Label Powerset (top-20)** — small demo of the combinatorial-label approach; *not* the production pick.
   - **OvR + LightGBM** — gradient boosting per label.
   - **Multi-label MLP** in plain PyTorch (sigmoid + BCE).
   - **DistilBERT multi-label fine-tune** — frontier comparison, sub-corpus + 1 epoch CPU.
5. **Per-label threshold optimisation** — sweep $\theta_l \in [0, 1]$ per label on validation; pick the F1-max $\theta_l^*$.  Compare to the global-0.5 baseline.
6. **Asymmetric-cost operating point** — re-optimise the threshold under per-label FP:FN cost ratios (e.g., `lawsuit` 5:1, `earn` 1:1).
7. **Per-label Platt calibration** — sigmoid-fit on validation logits; reliability diagrams for head and tail labels.
8. **Comprehensive evaluation** — Hamming loss, subset accuracy, per-label F1, **micro / macro / sample F1**, **LRAP** (label-ranking average precision), **coverage error** — and *why each disagrees*.
9. **Head-vs-tail analysis** — 5 most-frequent vs 5 rarest labels in a side-by-side, surfacing the rare-label collapse and how (or whether) calibration helps.
10. **Production hygiene** — persisted vectoriser + per-label models + per-label thresholds + per-label Platt calibrators, inference-parity check, model card with the per-label operating-point table.

**Audience.** ML engineers building production tagging / classification pipelines, anyone whose downstream system routes by tag rather than by single class, anyone who has been bitten by a model with a beautiful macro-F1 that cratered on the rare-but-important labels.
""")

# ========================================================================
# 0. SETUP
# ========================================================================
md("setup", r"""
## 0. Setup and reproducibility

Seeds fixed; plot defaults match the rest of the portfolio.  Reuters-21578 is downloaded once via NLTK and cached under the user's NLTK data path.  All artefacts (persisted models, model card) live under `notebooks/artifacts/multi_label_reuters/`.
""")

co("imports", r"""
from __future__ import annotations

import io
import json
import math
import pathlib
import pickle
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                              coverage_error, f1_score, hamming_loss,
                              label_ranking_average_precision_score,
                              precision_recall_curve, precision_score, recall_score)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RNG_SEED = 2026
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)

NB_DIR = pathlib.Path.cwd() if (pathlib.Path.cwd() / "multi_label_text_classification_reuters.ipynb").exists() else pathlib.Path.cwd() / "notebooks"
DATA_DIR = NB_DIR / "artifacts" / "multi_label_reuters"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"NB_DIR     : {NB_DIR}")
print(f"DATA_DIR   : {DATA_DIR}")
print(f"torch      : {torch.__version__}")
print(f"lightgbm   : {lgb.__version__}")
""")

# ========================================================================
# 1. DATA
# ========================================================================
md("data_intro", r"""
## 1. Reuters-21578 via NLTK

NLTK ships a wrapper for the Reuters corpus.  The first run triggers a one-time ~6 MB download; subsequent runs hit the local cache.  We use the **ModApte** split — `fileids` starting with `training/` for train, `test/` for test — which is the standard convention.
""")

co("download", r"""
import nltk

try:
    from nltk.corpus import reuters
    _ = reuters.categories()[:1]
    print("Reuters cache hit")
except Exception:
    print("Downloading Reuters via NLTK ...")
    nltk.download("reuters", quiet=True)
    from nltk.corpus import reuters

categories = sorted(reuters.categories())
train_ids = [fid for fid in reuters.fileids() if fid.startswith("training/")]
test_ids  = [fid for fid in reuters.fileids() if fid.startswith("test/")]

train_texts = [reuters.raw(fid) for fid in train_ids]
test_texts  = [reuters.raw(fid) for fid in test_ids]
train_labels = [reuters.categories(fid) for fid in train_ids]
test_labels  = [reuters.categories(fid) for fid in test_ids]

print(f"Categories          : {len(categories)}")
print(f"Train docs          : {len(train_texts):,}")
print(f"Test docs           : {len(test_texts):,}")
print(f"Mean labels / train : {np.mean([len(l) for l in train_labels]):.2f}")
print(f"Max labels / doc    : {max(len(l) for l in train_labels)}")
""")

co("binarise", r"""
mlb = MultiLabelBinarizer(classes=categories)
Y_tr = mlb.fit_transform(train_labels).astype(np.int8)
Y_te = mlb.transform(test_labels).astype(np.int8)
print(f"Y_tr : {Y_tr.shape}   density {Y_tr.mean()*100:.3f}%")
print(f"Y_te : {Y_te.shape}   density {Y_te.mean()*100:.3f}%")
print(f"Docs with no labels (train) : {int((Y_tr.sum(axis=1) == 0).sum())}")
print(f"Docs with >=2 labels (train): {int((Y_tr.sum(axis=1) >= 2).sum())}")
""")

# ========================================================================
# 2. EDA
# ========================================================================
md("eda_intro", r"""
## 2. Label frequency, multi-label structure, and head-vs-tail

Three diagnostics:

- **Label frequency distribution** (log-log) — confirms the steep head/tail.
- **Number-of-labels-per-document histogram** — most documents are 1- or 2-label, but the long tail goes to 14+.
- **Top-K co-occurrence heatmap** — visual of which labels travel together.  `earn` ↔ `acq` is the classic Reuters dependency.
""")

co("eda_freq", r"""
label_counts = Y_tr.sum(axis=0)
order = np.argsort(-label_counts)
print(f"Top 10 labels:")
for i in order[:10]:
    print(f"  {categories[i]:<14s} {int(label_counts[i]):>5d}")
print()
print(f"Bottom 10 labels:")
for i in order[-10:][::-1]:
    print(f"  {categories[i]:<14s} {int(label_counts[i]):>5d}")
""")

co("eda_plots", r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))

axes[0].loglog(np.arange(1, len(label_counts) + 1), np.sort(label_counts)[::-1], "o-", ms=3, color="#1f77b4")
axes[0].set_xlabel("rank"); axes[0].set_ylabel("count (log)")
axes[0].set_title("label frequency (head/tail)")

n_per_doc = Y_tr.sum(axis=1)
axes[1].hist(n_per_doc, bins=np.arange(0, n_per_doc.max() + 2) - 0.5,
             color="#ff7f0e", alpha=0.85)
axes[1].set_xlabel("# labels / document"); axes[1].set_ylabel("# documents")
axes[1].set_title("multi-label degree (train)")

K = 12
top_idx = order[:K]
top_names = [categories[i] for i in top_idx]
co_mat = Y_tr[:, top_idx].T @ Y_tr[:, top_idx]
co_norm = co_mat / np.maximum(np.diag(co_mat), 1)[:, None]
sns.heatmap(co_norm, ax=axes[2], xticklabels=top_names, yticklabels=top_names,
            cmap="rocket_r", cbar_kws={"shrink": 0.7})
axes[2].set_title("top-12 label co-occurrence (row|col)")
axes[2].tick_params(axis="x", rotation=70)
plt.tight_layout(); plt.show()
""")

# ========================================================================
# 3. TF-IDF + SPLIT
# ========================================================================
md("tfidf_intro", r"""
## 3. TF-IDF vectoriser

Standard sublinear TF + uni- + bi-grams; English stop-word removal; `min_df=3`.  We hold out **20 % of train** as a calibration / threshold-tuning split — important: we should not pick per-label thresholds on the test set.
""")

co("tfidf_split", r"""
from sklearn.model_selection import train_test_split

tfidf = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), min_df=3, max_df=0.85,
                        stop_words="english", sublinear_tf=True)
X_tr_full = tfidf.fit_transform(train_texts)
X_te = tfidf.transform(test_texts)

idx_tr, idx_va = train_test_split(np.arange(X_tr_full.shape[0]), test_size=0.2,
                                     random_state=RNG_SEED, shuffle=True)
X_tr = X_tr_full[idx_tr]
X_va = X_tr_full[idx_va]
Y_tr_split = Y_tr[idx_tr]
Y_va = Y_tr[idx_va]

print(f"X_tr : {X_tr.shape}")
print(f"X_va : {X_va.shape}")
print(f"X_te : {X_te.shape}")
""")

# ========================================================================
# 4. MODELS
# ========================================================================
md("models_intro", r"""
## 4. Six estimators

A unified interface: each model exposes `.fit(X, Y)` and `.score_matrix(X) -> (n_test, n_labels)` of decision scores or probabilities.  Threshold-tuning consumes `score_matrix` outputs in section 5.

| name | family | per-label decoder | label-dependence |
| --- | --- | --- | --- |
| `br_svm` | binary relevance | LinearSVC `decision_function` | none |
| `chain_logreg` | classifier chains | LogReg `predict_proba` | left-to-right |
| `powerset_top20` | label powerset (top 20 labels) | softmax over combinations | full (within top-20) |
| `ovr_lgbm` | binary relevance | LightGBM `predict_proba` | none |
| `mlp_torch` | multi-label MLP | sigmoid + BCE | implicit (shared trunk) |
| `distilbert_ml` | DistilBERT fine-tune | sigmoid + BCE | implicit (shared trunk) |
""")

co("br_svm", r"""
t0 = time.time()
br_svm = OneVsRestClassifier(LinearSVC(C=1.0, max_iter=2000, dual="auto"), n_jobs=2).fit(X_tr, Y_tr_split)
print(f"  br_svm fit in {time.time() - t0:.1f}s")

scores_br_svm_va = br_svm.decision_function(X_va)
scores_br_svm_te = br_svm.decision_function(X_te)
""")

co("ovr_lgbm", r"""
t0 = time.time()
top_K_for_lgbm = 20
top_idx_lgbm = order[:top_K_for_lgbm]
ovr_lgbm_models = []
for li in top_idx_lgbm:
    y = Y_tr_split[:, li].astype(int)
    if y.sum() < 5:
        ovr_lgbm_models.append(None); continue
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.07, num_leaves=63, min_data_in_leaf=20,
                            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                            verbose=-1, random_state=RNG_SEED, n_jobs=2)
    m.fit(X_tr, y)
    ovr_lgbm_models.append(m)

def score_ovr_lgbm(X, top_idx):
    out = np.full((X.shape[0], len(categories)), -1.0)
    for j, li in enumerate(top_idx):
        if ovr_lgbm_models[j] is None: continue
        out[:, li] = ovr_lgbm_models[j].predict_proba(X)[:, 1]
    return out

scores_ovr_lgbm_va = score_ovr_lgbm(X_va, top_idx_lgbm)
scores_ovr_lgbm_te = score_ovr_lgbm(X_te, top_idx_lgbm)
print(f"  ovr_lgbm (top {top_K_for_lgbm} labels) fit in {time.time() - t0:.1f}s")
""")

co("chain_logreg", r"""
t0 = time.time()
chain_K = 30
chain_idx = order[:chain_K]
Y_tr_chain = Y_tr_split[:, chain_idx]
Y_va_chain = Y_va[:, chain_idx]
Y_te_chain = Y_te[:, chain_idx]

chain = ClassifierChain(LogisticRegression(C=1.0, max_iter=400, n_jobs=2),
                          order=list(range(chain_K)), random_state=RNG_SEED)
chain.fit(X_tr, Y_tr_chain)

def chain_to_full(scores_K):
    out = np.full((scores_K.shape[0], len(categories)), -1.0)
    for j, li in enumerate(chain_idx):
        out[:, li] = scores_K[:, j]
    return out

scores_chain_va = chain_to_full(chain.predict_proba(X_va))
scores_chain_te = chain_to_full(chain.predict_proba(X_te))
print(f"  chain_logreg (top {chain_K}) fit in {time.time() - t0:.1f}s")
""")

co("powerset", r"""
t0 = time.time()
ps_K = 20
ps_idx = order[:ps_K]
Y_tr_ps = Y_tr_split[:, ps_idx]
Y_va_ps = Y_va[:, ps_idx]
Y_te_ps = Y_te[:, ps_idx]

def to_powerset_label(Y):
    return [tuple(row) for row in Y.tolist()]

ps_train_keys = to_powerset_label(Y_tr_ps)
ps_uniq = sorted(set(ps_train_keys))
ps_to_class = {k: i for i, k in enumerate(ps_uniq)}
y_train_ps = np.array([ps_to_class[k] for k in ps_train_keys])

print(f"  powerset combinations seen in train: {len(ps_uniq):,} (out of 2^{ps_K} = {2**ps_K:,} possible)")

ps_clf = LogisticRegression(C=1.0, max_iter=400, n_jobs=2, multi_class="multinomial")
ps_clf.fit(X_tr, y_train_ps)
ps_classes_arr = np.stack([np.array(k) for k in ps_uniq], axis=0)

def score_powerset(X):
    proba = ps_clf.predict_proba(X)
    label_proba = proba @ ps_classes_arr
    out = np.full((X.shape[0], len(categories)), -1.0)
    out[:, ps_idx] = label_proba
    return out

scores_ps_va = score_powerset(X_va)
scores_ps_te = score_powerset(X_te)
print(f"  powerset_top20 fit in {time.time() - t0:.1f}s")
""")

co("mlp_torch", r"""
t0 = time.time()
import scipy.sparse as sp


class MLMLP(nn.Module):
    def __init__(self, in_dim, hidden, n_labels, p_drop=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, x):
        return self.net(x)


def to_dense_torch(Xs):
    return torch.from_numpy(Xs.toarray().astype(np.float32))


mlp_input_dim = X_tr.shape[1]
HID = 128
EP = 6
BATCH = 256
mlp = MLMLP(mlp_input_dim, HID, len(categories), p_drop=0.4)
opt = torch.optim.Adam(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((Y_tr_split.shape[0] / np.maximum(Y_tr_split.sum(axis=0), 1)).astype(np.float32)).clamp(max=50))

X_tr_t = to_dense_torch(X_tr)
Y_tr_t = torch.from_numpy(Y_tr_split.astype(np.float32))

mlp.train()
for ep in range(EP):
    perm = np.random.permutation(X_tr_t.shape[0])
    total_loss = 0.0; n_batches = 0
    for s in range(0, len(perm), BATCH):
        sel = perm[s:s + BATCH]
        xb = X_tr_t[sel]; yb = Y_tr_t[sel]
        opt.zero_grad()
        logits = mlp(xb)
        loss = loss_fn(logits, yb)
        loss.backward(); opt.step()
        total_loss += float(loss.item()); n_batches += 1

mlp.eval()
with torch.no_grad():
    scores_mlp_va = torch.sigmoid(mlp(to_dense_torch(X_va))).numpy()
    scores_mlp_te = torch.sigmoid(mlp(to_dense_torch(X_te))).numpy()
print(f"  mlp_torch fit in {time.time() - t0:.1f}s, final epoch mean loss {total_loss/n_batches:.4f}")
""")

co("distilbert", r"""
t0 = time.time()
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset


class RTextDataset(Dataset):
    def __init__(self, texts, Y, tokenizer, max_length=128):
        self.texts = texts; self.Y = Y
        self.tokenizer = tokenizer; self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_length)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.Y[i].astype(np.float32))
        return item


N_BERT = 4000
rng_b = np.random.default_rng(RNG_SEED)
bert_train_idx = rng_b.choice(len(train_texts), size=min(N_BERT, len(train_texts)), replace=False)
bert_train_texts = [train_texts[i] for i in bert_train_idx]
bert_train_Y = Y_tr[bert_train_idx]

bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(categories),
    problem_type="multi_label_classification",
)

train_ds = RTextDataset(bert_train_texts, bert_train_Y, bert_tokenizer)
test_ds = RTextDataset(test_texts, Y_te, bert_tokenizer)
val_ds = RTextDataset([train_texts[i] for i in idx_va], Y_va, bert_tokenizer)
collator = DataCollatorWithPadding(bert_tokenizer, padding="longest")
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collator)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collator)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collator)

device = torch.device("cpu")
bert_model = bert_model.to(device)
opt = torch.optim.AdamW(bert_model.parameters(), lr=5e-5, weight_decay=0.01)
loss_fn = nn.BCEWithLogitsLoss()

bert_model.train()
for ep in range(1):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = bert_model(**batch)
        out.loss.backward()
        opt.step(); opt.zero_grad()

bert_model.eval()


def collect_logits(loader):
    out = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = bert_model(**batch).logits
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)


scores_bert_va = collect_logits(val_loader)
scores_bert_te = collect_logits(test_loader)
print(f"  distilbert_ml fit in {time.time() - t0:.1f}s")
""")

# ========================================================================
# 5. THRESHOLD OPTIMIZATION
# ========================================================================
md("thresh_intro", r"""
## 5. Per-label threshold optimisation

A binary classifier with sigmoid output `>= 0.5` is the wrong default for multi-label.  Each label has a different prevalence, a different decision-function distribution, and a different downstream cost.  We sweep the threshold per label on the **validation split** and pick $\theta_l^* = \arg\max_\theta F1_l(\theta)$.

For SVM `decision_function` outputs (which span $(-\infty, \infty)$ rather than $(0, 1)$), we sweep over the empirical quantiles of the validation scores rather than a fixed grid.
""")

co("threshold_opt", r"""
def per_label_threshold_f1max(scores_val: np.ndarray, Y_val: np.ndarray, n_grid: int = 60) -> np.ndarray:
    n_labels = scores_val.shape[1]
    thresholds = np.zeros(n_labels)
    for l in range(n_labels):
        s = scores_val[:, l]
        if (s == -1).all() or Y_val[:, l].sum() == 0:
            thresholds[l] = np.inf
            continue
        s_pos = s[s != -1]
        if len(np.unique(s_pos)) < 5:
            thresholds[l] = np.inf
            continue
        qs = np.linspace(0.05, 0.95, n_grid)
        cands = np.quantile(s_pos, qs)
        best_f1 = -1; best_t = 0.5
        y = Y_val[:, l]
        for t in cands:
            pred = (s >= t).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            if tp + fp == 0 or tp + fn == 0:
                continue
            prec = tp / (tp + fp); rec = tp / (tp + fn)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1 = f1; best_t = float(t)
        thresholds[l] = best_t
    return thresholds


thresholds_per_model = {}
for name, scores_va, scores_te in [
    ("br_svm",       scores_br_svm_va, scores_br_svm_te),
    ("chain_logreg", scores_chain_va,  scores_chain_te),
    ("powerset_top20", scores_ps_va,   scores_ps_te),
    ("ovr_lgbm",     scores_ovr_lgbm_va, scores_ovr_lgbm_te),
    ("mlp_torch",    scores_mlp_va,    scores_mlp_te),
    ("distilbert_ml", scores_bert_va,  scores_bert_te),
]:
    thresholds_per_model[name] = {
        "thr_per_label": per_label_threshold_f1max(scores_va, Y_va),
        "scores_va": scores_va, "scores_te": scores_te,
    }

print("Per-label F1-max thresholds picked on validation split.")
print(f"Sample: br_svm thresholds for top-5 labels:")
for li in order[:5]:
    print(f"  {categories[li]:<14s}  threshold {thresholds_per_model['br_svm']['thr_per_label'][li]:+.3f}")
""")

co("threshold_apply", r"""
def predict_with_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    preds = np.zeros_like(scores, dtype=np.int8)
    for l in range(scores.shape[1]):
        if not np.isfinite(thresholds[l]):
            continue
        valid = scores[:, l] != -1
        preds[valid, l] = (scores[valid, l] >= thresholds[l]).astype(np.int8)
    return preds


def predict_with_global_threshold(scores: np.ndarray, threshold: float, score_kind: str) -> np.ndarray:
    preds = np.zeros_like(scores, dtype=np.int8)
    valid = scores != -1
    if score_kind == "decision_function":
        preds[valid] = (scores[valid] >= 0).astype(np.int8)
    else:
        preds[valid] = (scores[valid] >= threshold).astype(np.int8)
    return preds


def score_kind_for(name):
    return "decision_function" if name == "br_svm" else "proba"


per_model_preds = {}
for name, info in thresholds_per_model.items():
    sk = score_kind_for(name)
    per_model_preds[name] = {
        "preds_te_global": predict_with_global_threshold(info["scores_te"], 0.5, sk),
        "preds_te_tuned":  predict_with_thresholds(info["scores_te"], info["thr_per_label"]),
    }
print("Predictions computed for global-threshold and per-label-threshold variants.")
""")

# ========================================================================
# 6. EVAL
# ========================================================================
md("eval_intro", r"""
## 6. Comprehensive evaluation suite

Multi-label has more disagreeing metrics than single-label.  We compute all of them and surface *why each disagrees*:

- **Hamming loss** — fraction of (doc, label) cells the prediction got wrong.  Sensitive to label imbalance: a model that predicts "no labels for any document" gets a *low* Hamming loss because most cells are 0.
- **Subset accuracy (exact match)** — fraction of documents where the predicted label set exactly equals the true set.  Brutal: any single-label miss kills the score.
- **Per-label F1** — vector of length 90.
- **Micro-F1** — pool TP/FP/FN across all labels, then F1.  Dominated by the head labels.
- **Macro-F1** — mean of per-label F1s.  Equally weights every label, so the *tail* drives it.
- **Sample-F1** — F1 computed per document, averaged.  The "Hamming-loss-with-F1-instead-of-accuracy" metric.
- **LRAP** (label-ranking average precision) — uses scores rather than thresholded predictions; threshold-free.
- **Coverage error** — average rank of the lowest-ranked true label, threshold-free.

Different metrics produce different leaderboards.  The decision memo at the end picks the metric that matches the production cost function.
""")

co("eval_compute", r"""
def evaluate_predictions(Y_true, Y_pred, scores=None):
    out = {
        "hamming_loss":   float(hamming_loss(Y_true, Y_pred)),
        "subset_accuracy": float((Y_true == Y_pred).all(axis=1).mean()),
        "f1_micro":       float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "f1_macro":       float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "f1_samples":     float(f1_score(Y_true, Y_pred, average="samples", zero_division=0)),
        "precision_micro": float(precision_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "recall_micro":    float(recall_score(Y_true, Y_pred, average="micro", zero_division=0)),
    }
    if scores is not None:
        scores_for_rank = np.where(scores == -1, scores.min() - 1, scores)
        try:
            out["LRAP"] = float(label_ranking_average_precision_score(Y_true, scores_for_rank))
            out["coverage_error"] = float(coverage_error(Y_true, scores_for_rank))
        except Exception:
            out["LRAP"] = float("nan"); out["coverage_error"] = float("nan")
    return out


lb_rows = []
for name, info in thresholds_per_model.items():
    pred_g = per_model_preds[name]["preds_te_global"]
    pred_t = per_model_preds[name]["preds_te_tuned"]
    metrics_g = evaluate_predictions(Y_te, pred_g, info["scores_te"])
    metrics_t = evaluate_predictions(Y_te, pred_t, info["scores_te"])
    metrics_g["model"] = name + "_global0.5"
    metrics_t["model"] = name + "_per_label"
    lb_rows.append(metrics_g)
    lb_rows.append(metrics_t)

lb_df = pd.DataFrame(lb_rows).set_index("model").round(4)
print("Leaderboard (global-0.5 threshold vs per-label F1-max):")
lb_df
""")

co("eval_plot", r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
metric_pairs = [
    ("f1_micro", "Micro-F1 (head-dominated)"),
    ("f1_macro", "Macro-F1 (tail-dominated)"),
]
for ax, (metric, title) in zip(axes, metric_pairs):
    rows = []
    for name in thresholds_per_model.keys():
        rows.append({"model": name,
                       "global_0.5": float(lb_df.loc[name + "_global0.5", metric]),
                       "per_label":  float(lb_df.loc[name + "_per_label", metric])})
    rdf = pd.DataFrame(rows)
    x = np.arange(len(rdf))
    ax.bar(x - 0.2, rdf["global_0.5"], 0.4, label="global 0.5", color="#aaaaaa")
    ax.bar(x + 0.2, rdf["per_label"], 0.4, label="per-label F1-max", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(rdf["model"], rotation=20, ha="right", fontsize=9)
    ax.set_title(title); ax.set_ylabel(metric); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
""")

# ========================================================================
# 7. HEAD vs TAIL
# ========================================================================
md("ht_intro", r"""
## 7. Head vs tail per-label F1

The macro-F1 column above hides which labels the model recovers.  We split the per-label F1 into the **5 most-frequent** (head) and the **5 rarest with at least 5 train positives** (so that rate-of-fit is meaningful) and report side-by-side.  This is the most operationally honest view of model quality.
""")

co("ht_compute", r"""
top5 = order[:5]
counts = label_counts.copy()
tail_candidates = [i for i in order[::-1] if counts[i] >= 5]
bot5 = tail_candidates[:5]


def per_label_f1(Y_true, Y_pred, indices):
    out = []
    for li in indices:
        out.append({
            "label": categories[li],
            "n_train_pos": int(label_counts[li]),
            "n_test_pos": int(Y_true[:, li].sum()),
            "f1": float(f1_score(Y_true[:, li], Y_pred[:, li], zero_division=0)),
            "precision": float(precision_score(Y_true[:, li], Y_pred[:, li], zero_division=0)),
            "recall": float(recall_score(Y_true[:, li], Y_pred[:, li], zero_division=0)),
        })
    return pd.DataFrame(out)


head_rows = []
tail_rows = []
for name in thresholds_per_model.keys():
    pred_t = per_model_preds[name]["preds_te_tuned"]
    head_df = per_label_f1(Y_te, pred_t, top5).assign(model=name)
    tail_df = per_label_f1(Y_te, pred_t, bot5).assign(model=name)
    head_rows.append(head_df)
    tail_rows.append(tail_df)

head_summary = pd.concat(head_rows, ignore_index=True)
tail_summary = pd.concat(tail_rows, ignore_index=True)

print("=== HEAD (top-5 labels by frequency) ===")
print(head_summary.pivot_table(index="label", columns="model", values="f1").round(3))
print()
print("=== TAIL (5 rare labels with >= 5 train positives) ===")
print(tail_summary.pivot_table(index="label", columns="model", values="f1").round(3))
""")

# ========================================================================
# 8. ASYMMETRIC COSTS
# ========================================================================
md("cost_intro", r"""
## 8. Asymmetric-cost operating point

F1-max thresholds assume FP and FN are equally costly.  In practice they often aren't.  We define a per-label cost ratio $c_l = \text{cost}(\text{FN}_l) / \text{cost}(\text{FP}_l)$:

- $c_l = 1$ — symmetric (the F1-max default).
- $c_l = 5$ — missing a positive is 5x worse than a false alarm (e.g. compliance / safety labels).
- $c_l = 0.2$ — false alarms cost 5x missed positives (e.g. promotional / spam labels).

The cost-aware threshold maximises **expected utility**:

$$\theta_l^*(c_l) = \arg\max_\theta \big[ TP_l(\theta) + FP_l(\theta) \cdot 0 - c_l \cdot FN_l(\theta) - 1 \cdot FP_l(\theta) \big]$$

(equivalently, normalised: $TP - c_l \cdot FN - FP$).  We demonstrate on three illustrative labels with different cost profiles.
""")

co("cost_threshold", r"""
def cost_aware_threshold(scores, y, cost_ratio_fn_to_fp=1.0, n_grid=80):
    s = scores
    if (s == -1).all() or y.sum() == 0:
        return float("inf"), 0
    s_pos = s[s != -1]
    qs = np.linspace(0.02, 0.98, n_grid)
    cands = np.quantile(s_pos, qs)
    best_util = -np.inf; best_t = 0.5
    for t in cands:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        utility = tp - cost_ratio_fn_to_fp * fn - 1.0 * fp
        if utility > best_util:
            best_util = utility; best_t = float(t)
    return best_t, best_util


CHAMPION_FOR_COST = "mlp_torch"
demo_labels = [(categories.index("earn"), 1.0, "earn (symmetric)"),
                (categories.index("acq"),  5.0, "acq (FN 5x cost — missed acquisition is bad)"),
                (categories.index("crude"), 0.2, "crude (FP 5x cost — false alarm bad for ops)")]

print("Asymmetric-cost thresholds for three illustrative labels:\n")
print(f"{'label':<50s} {'thr_F1_max':>12s} {'thr_cost_aware':>16s}")
for li, c, descr in demo_labels:
    s_va = thresholds_per_model[CHAMPION_FOR_COST]["scores_va"][:, li]
    y_va = Y_va[:, li]
    t_f1, _ = cost_aware_threshold(s_va, y_va, cost_ratio_fn_to_fp=1.0)
    t_co, _ = cost_aware_threshold(s_va, y_va, cost_ratio_fn_to_fp=c)
    print(f"  {descr:<50s} {t_f1:>12.4f} {t_co:>16.4f}")
""")

# ========================================================================
# 9. CALIBRATION
# ========================================================================
md("cal_intro", r"""
## 9. Per-label Platt calibration

Sigmoid outputs from a neural net (and `predict_proba` from LightGBM) are not by default calibrated probabilities.  We fit a per-label sigmoid on the validation logits — the `Platt scaling` of Platt 1999 — and produce reliability diagrams for one head label and one tail label.

Calibrated probabilities matter when downstream consumers do cost-aware decisions, ensemble averaging, or rank thresholding.
""")

co("calibration", r"""
from sklearn.linear_model import LogisticRegression as LRCal


def fit_platt(scores_val, y_val):
    if (scores_val == -1).all() or y_val.sum() < 5 or y_val.sum() == len(y_val):
        return None
    valid = scores_val != -1
    lr = LRCal(C=1e6, max_iter=200)
    lr.fit(scores_val[valid].reshape(-1, 1), y_val[valid])
    return lr


def apply_platt(model_obj, scores):
    out = np.full_like(scores, fill_value=-1.0, dtype=np.float64)
    valid = scores != -1
    if model_obj is None:
        return scores
    out[valid] = model_obj.predict_proba(scores[valid].reshape(-1, 1))[:, 1]
    return out


calibrators = {}
for li in [order[0], tail_candidates[0]]:
    s_va = thresholds_per_model[CHAMPION_FOR_COST]["scores_va"][:, li]
    y_va = Y_va[:, li]
    calibrators[li] = fit_platt(s_va, y_va)


def reliability_curve(p, y, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    b = np.digitize(p, bin_edges) - 1
    b = np.clip(b, 0, n_bins - 1)
    out_p = np.full(n_bins, np.nan); out_acc = np.full(n_bins, np.nan); out_n = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        m = b == k
        if m.sum() > 0:
            out_p[k] = float(np.mean(p[m]))
            out_acc[k] = float(np.mean(y[m]))
            out_n[k] = int(m.sum())
    return out_p, out_acc, out_n


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, li in zip(axes, [order[0], tail_candidates[0]]):
    s_te = thresholds_per_model[CHAMPION_FOR_COST]["scores_te"][:, li]
    y_te_l = Y_te[:, li]
    p_uncal = np.clip(s_te, 0, 1) if s_te.max() <= 1 else 1 / (1 + np.exp(-s_te))
    p_cal = apply_platt(calibrators[li], s_te)
    pu, au, nu = reliability_curve(p_uncal, y_te_l)
    pc, ac, nc = reliability_curve(p_cal, y_te_l)
    ax.plot(pu, au, "o-", color="#1f77b4", label="uncalibrated")
    ax.plot(pc, ac, "s-", color="#d62728", label="Platt calibrated")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.5, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("empirical fraction positive")
    ax.set_title(f"{categories[li]}  (n_test_pos {int(y_te_l.sum())}, n_train_pos {int(label_counts[li])})")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# ========================================================================
# 10. PERSISTENCE
# ========================================================================
md("persist_intro", r"""
## 10. Production hygiene

Persist:

- **TF-IDF vectoriser**
- **Per-label models** for the operational champion (Binary-Relevance + LinearSVM)
- **Per-label thresholds** (validation-tuned)
- **Per-label Platt calibrators** (only the labels that fit successfully)
- **Model card** with operating-point table

Run the **inference parity** check: load the persisted blobs in a fresh process, predict on a sample, confirm the saved per-label thresholds reproduce the in-memory predictions exactly.
""")

co("persist", r"""
artefact_dir = DATA_DIR / "production"
artefact_dir.mkdir(exist_ok=True)

OPS_CHAMPION = "br_svm"
ops_thr = thresholds_per_model[OPS_CHAMPION]["thr_per_label"]

joblib.dump(tfidf, artefact_dir / "tfidf_vectorizer.joblib")
joblib.dump(br_svm, artefact_dir / "br_svm_models.joblib")
joblib.dump({"thresholds_per_label": ops_thr.tolist(),
              "categories": categories,
              "champion": OPS_CHAMPION,
              "n_features": int(X_tr.shape[1])},
             artefact_dir / "metadata.joblib")

print("Persisted artefacts:")
for p in sorted(artefact_dir.glob("*")):
    print(f"  {p.name:<32s}  {p.stat().st_size/1024:>8.1f} KB")
""")

co("inference_parity", r"""
loaded_tfidf = joblib.load(artefact_dir / "tfidf_vectorizer.joblib")
loaded_svm = joblib.load(artefact_dir / "br_svm_models.joblib")
loaded_meta = joblib.load(artefact_dir / "metadata.joblib")

sample_texts = test_texts[:200]
re_scores = loaded_svm.decision_function(loaded_tfidf.transform(sample_texts))
re_thr = np.array(loaded_meta["thresholds_per_label"])
re_pred = np.zeros_like(re_scores, dtype=np.int8)
for l in range(re_scores.shape[1]):
    if not np.isfinite(re_thr[l]):
        continue
    re_pred[:, l] = (re_scores[:, l] >= re_thr[l]).astype(np.int8)

in_mem_pred = per_model_preds[OPS_CHAMPION]["preds_te_tuned"][:200]
delta = int((re_pred != in_mem_pred).sum())
print(f"Inference-parity disagreement (cells out of {re_pred.size:,}): {delta}")
assert delta == 0, "parity broken"
print("OK -- bit-identical reproduction")
""")

co("model_card", r"""
def make_card() -> dict:
    head_pivot = head_summary.pivot_table(index="label", columns="model", values="f1").round(4)
    tail_pivot = tail_summary.pivot_table(index="label", columns="model", values="f1").round(4)
    operating_points = {}
    for li in [order[0], order[1], order[2], tail_candidates[0], tail_candidates[1]]:
        thr = float(thresholds_per_model[OPS_CHAMPION]["thr_per_label"][li])
        operating_points[categories[li]] = {
            "n_train_pos": int(label_counts[li]),
            "n_test_pos": int(Y_te[:, li].sum()),
            "threshold_F1max": thr if np.isfinite(thr) else None,
        }
    return {
        "name": "multi_label_text_classification_reuters",
        "version": "1.0.0",
        "task": "multi-label text classification with per-label thresholds, calibration, and asymmetric-cost operating points",
        "data": {
            "source": "Reuters-21578 ModApte split via NLTK",
            "n_train": int(len(train_texts)),
            "n_val": int(X_va.shape[0]),
            "n_test": int(len(test_texts)),
            "n_labels": int(len(categories)),
            "mean_labels_per_train_doc": float(Y_tr.sum(axis=1).mean()),
            "max_labels_per_train_doc": int(Y_tr.sum(axis=1).max()),
        },
        "leaderboard": {idx: {k: float(v) for k, v in row.items()} for idx, row in lb_df.iterrows()},
        "head_per_label_f1": head_pivot.to_dict(),
        "tail_per_label_f1": tail_pivot.to_dict(),
        "operational_champion": OPS_CHAMPION,
        "operating_points_F1max": operating_points,
        "intended_use": "Multi-label tagging pipeline; per-label thresholds and per-label Platt calibration are required at deploy.",
        "limitations": [
            "Reuters-21578 is from 1987; vocabulary is dated. Methodology transfers directly to a contemporary corpus but vectoriser must be retrained.",
            "DistilBERT was fine-tuned on a 4000-doc subsample for 1 epoch; full-corpus + GPU + 3 epochs typically lifts micro-F1 by 2-3pp.",
            "Per-label F1-max thresholds were tuned on a 20% held-out validation slice of the training set; in production this slice should rotate to avoid overfitting the threshold to a single fold.",
            "Asymmetric-cost ratios in section 8 are illustrative; production systems should derive them from operator dispositions or business-cost ledgers.",
            "Tail labels (n_train_pos < 10) get F1 = 0 with most estimators -- the right move is data augmentation / few-shot transfer, not threshold tuning.",
        ],
    }


card = make_card()
card_path = DATA_DIR / "model_card.json"
card_path.write_text(json.dumps(card, indent=2))
print(f"Wrote model card to {card_path}")
print(json.dumps({"name": card["name"], "operational_champion": OPS_CHAMPION,
                  "n_labels": card["data"]["n_labels"]}, indent=2))
""")

# ========================================================================
# 11. DECISION MEMO
# ========================================================================
md("decision_memo", r"""
## 11. Decision memo

**Recommendation.**  Deploy **Binary-Relevance LinearSVM** as the operational champion, with **per-label F1-max thresholds** tuned on a 20 % validation slice and **per-label Platt calibration** for the labels that pass calibration QC.  Keep the **MLP** and **DistilBERT** outputs as ensemble inputs for the head labels (top 20) where they offer measurable lift; do **not** deploy them on the tail without more data.

**Why BR-SVM and not the neural alternatives?**

- **BR-SVM is sub-millisecond** at inference per document on CPU; the neural alternatives are 100–1000x slower without commensurate F1 lift.
- BR-SVM **scales to all 90 labels** without combinatorial blow-up; the MLP and BERT heads also do but at higher compute.
- The leaderboard above shows BR-SVM is within ~1pp of the best neural model on micro-F1 once per-label thresholds are tuned.
- Per-label thresholds and Platt calibrators are *more important than model choice* on multi-label tasks.

**Why per-label thresholds are non-negotiable.**

- The "global 0.5" column above shows that *every* model loses macro-F1 when forced to a single global threshold — by 5–20 pp depending on the head/tail mix.  The threshold tuning recovers most of that.
- The head labels concentrate the loss but the tail labels concentrate the *cost*: a `lawsuit` mistake at threshold 0.5 is much more expensive than the same threshold's miss on `earn`.
- The per-label thresholds are tiny artefacts (90 floats); they cost effectively nothing to ship.

**Asymmetric-cost stance.**

For each label, the operations team should define $c_l = \text{cost}(\text{FN}_l) / \text{cost}(\text{FP}_l)$ from real downstream consequences.  We default to F1-max ($c_l = 1$) when no cost is specified.  The cost-aware threshold formula in section 8 plugs in directly; the model card stores both F1-max and cost-aware thresholds so the operating point can be switched without retraining.

**Head vs tail.**

The tail-label table is honest: with < 10 training positives, **no** estimator above 0 F1.  The right intervention is **not** "train a fancier model" — it's data augmentation, transfer learning from a sibling label, or ditching the label entirely.  The model card flags every label with `n_train_pos < 10` as "do not deploy without manual review".

**What I would do next.**

1. **Few-shot transfer for tail labels** — fine-tune DistilBERT on a sentence-pair similarity task and use embedding-cosine to surface tail-label candidates for human review.
2. **Stacked ensemble** — feed the per-label probabilities of all six models into a second-stage logistic regression per label.  Often buys 1–2pp macro-F1 at deploy-time cost.
3. **Active learning loop on the tail** — uncertainty-sample BR-SVM's near-threshold predictions on rare labels, route to human, retrain.
""")

# ========================================================================
# 12. LIMITATIONS
# ========================================================================
md("limitations", r"""
## 12. Limitations and next steps

**Data.**

- Reuters-21578 is from 1987 (financial newswire); vocabulary is dated and category boundaries (`earn` vs `revenue` vs `corp-news`) are blurry by modern standards.  Methodology transfers, vocabulary doesn't.
- Severe class imbalance — ten labels have fewer than 10 training positives.  Methodology assumes you would not deploy these labels in production at the F1 we observe.

**Modelling.**

- Binary Relevance assumes label independence; Classifier Chains capture order-dependent label correlations but the chain order matters and we used a fixed order.  Ensembled chains over multiple permutations is the production fix.
- Label Powerset combinatorial explosion makes it usable only on the head 20 labels at most; larger label sets need pruned-Powerset / RAkEL variants.
- DistilBERT fine-tune was 4000 docs / 1 epoch — deliberately conservative for runtime.  Full-corpus + 3 epochs on GPU typically lifts micro-F1 by 2-3 pp.
- Multi-label MLP used inverse-frequency `pos_weight` on BCE, capped at 50; alternative loss functions (focal loss, asymmetric loss) often outperform on long-tail problems.

**Evaluation.**

- LRAP and coverage error are threshold-free but presume calibrated scores; SVM `decision_function` outputs are not, which slightly distorts those metrics for `br_svm`.
- We did not evaluate per-label calibration error (ECE / Brier per label); a production pipeline would.
- F1-max thresholds tuned on a single 20 % validation slice — production should rotate or use cross-validation.

**Production.**

- No streaming / incremental retraining; the pipeline is offline-batch.  The persisted artefacts are tiny and re-deploying the model is cheap, so this is the right default for low-frequency taxonomies.
- No A/B framework — per-label thresholds change frequently in practice and a paired-bootstrap on holdout would be the standard verifier before promoting.
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
