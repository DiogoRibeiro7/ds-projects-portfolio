"""Builds notebooks/geo_experiments_hierarchical_bayesian.ipynb.

Run from repo root:  python _build_geo_nb.py
"""
from __future__ import annotations

import hashlib
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "notebooks" / "geo_experiments_hierarchical_bayesian.ipynb"

cells: list = []


def _cid(kind: str, key: str) -> str:
    h = hashlib.md5(f"{kind}-{key}".encode()).hexdigest()[:6]
    return f"geo-{kind}-{key.replace('_', '-')}-{h}"


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
# Geo Experiments and Hierarchical-Bayesian Meta-Analysis: Synthetic Control + Random-Effects Pooling

**The problem.** A consumer-brand marketing team controls a quarterly ad budget across 30 metro markets.  Standard A/B testing is infeasible — exposure cannot be randomised at the user level (TV / out-of-home / billboards reach everyone in a market) — so they run **geo experiments**: a subset of markets receives the spend increase, the rest do not, and the lift on weekly sales is the causal estimand.  Three answers are needed:

1. **Per-market lift** with a 95 % CI — for each treated market, did the spend deliver?
2. **Global meta-analytic lift** with a credible interval — what should the planner expect from a similar campaign next quarter?
3. **Heterogeneity** — how much do market-level effects differ from the global mean?  This determines whether the next planning cycle should treat every market the same (small $\tau$) or invest in market-specific creative ($\tau$ large).

A naive *pre / post* comparison is contaminated by trend, seasonality, and concurrent shocks; a *difference-in-differences* helps but assumes parallel trends; **synthetic control** (Abadie 2003) is the workhorse but produces only point estimates per market with bootstrap CIs that ignore cross-market information.  The right answer pools the per-market synthetic-control estimates **hierarchically**: $\theta_i \sim \mathcal{N}(\mu, \tau^2)$ — random-effects meta-analysis (DerSimonian & Laird).  The hierarchical model **shrinks** noisy market estimates toward the global mean, produces **calibrated CIs**, and gives the planner **both** a per-market and a global view in one inference.

**The data.** Simulated.  We control the data-generating process so we have **ground-truth heterogeneous lifts $\theta_i$** for each treated market — and can therefore measure estimator bias, coverage, and shrinkage *honestly*, which is impossible on real geo data.  The methodology transfers directly to real Nielsen-style market panel datasets.

**The approach.**

1. **Simulate** 30 markets × 104 weeks, with realistic trend, seasonality, and idiosyncratic noise; randomly assign 10 markets to treatment in week 53; planted lifts $\theta_i \sim \mathcal{N}(0.05, 0.02)$ on a 5 %-of-baseline scale.
2. **EDA** — per-market sales paths, treatment-vs-control distinction, pre-period balance check.
3. **Estimator lineup**:
   - **Naive pre / post** — the obvious wrong thing; baseline for what bias looks like.
   - **DiD** (Difference-in-Differences) — basic causal baseline; assumes parallel trends.
   - **Synthetic Control** — Abadie-style donor-pool weighted average, hand-rolled with a constrained-OLS solve, bootstrap CIs per treated market.
   - **TBR / Bayesian structural time series** — state-space sketch on one market for an alternative posterior view.
4. **Hierarchical Bayesian meta-analysis** — feed the per-market synthetic-control estimates $(\hat\theta_i, \sigma_i)$ to a random-effects model
   $$\theta_i \sim \mathcal{N}(\mu, \tau^2), \quad \hat\theta_i \sim \mathcal{N}(\theta_i, \sigma_i^2)$$
   with a hand-rolled **Metropolis-Hastings** MCMC sampler (matches the fisheries / SEIR pattern in the rest of the portfolio — no PyMC / NumPyro dependency).
5. **Calibration and diagnostics**:
   - Coverage: at the 95 % nominal level, what fraction of CIs contain the ground truth?
   - Bias / RMSE per estimator.
   - **Forest plot** of per-market estimates with the meta-analytic global posterior.
   - **Posterior predictive checks** for the hierarchical model.
   - **MCMC diagnostics**: trace plots, R-hat across chains, effective sample size.
6. **Power / MDE curve** for the next planning cycle.
7. **Decision memo** — which markets to expand, which to drop, and what the global expected lift implies for next quarter's budget.
8. **Production hygiene** — persisted MCMC samples, posterior summary tables, model card.

**Audience.** Marketing-mix / measurement teams, public-policy analysts running quasi-experiments, anyone whose A/B test is forced to be at a *region* rather than a *user* — and anyone who has felt that a single point estimate per market is leaving information on the table.
""")

# ========================================================================
# 0. SETUP
# ========================================================================
md("setup", r"""
## 0. Setup and reproducibility

Seeds fixed; plot defaults match the rest of the portfolio.  All artefacts (MCMC samples, posterior summaries, model card) live under `notebooks/artifacts/geo_experiments/`.
""")

co("imports", r"""
from __future__ import annotations

import io
import json
import math
import pathlib
import time
import warnings
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RNG_SEED = 2026
np.random.seed(RNG_SEED)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)

NB_DIR = pathlib.Path.cwd() if (pathlib.Path.cwd() / "geo_experiments_hierarchical_bayesian.ipynb").exists() else pathlib.Path.cwd() / "notebooks"
DATA_DIR = NB_DIR / "artifacts" / "geo_experiments"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"NB_DIR    : {NB_DIR}")
print(f"DATA_DIR  : {DATA_DIR}")
print(f"numpy     : {np.__version__}")
print(f"scipy     : {__import__('scipy').__version__}")
""")

# ========================================================================
# 1. SIMULATE
# ========================================================================
md("sim_intro", r"""
## 1. Data-generating process

30 markets, 104 weekly observations.  Each market $m$ has:

- **Baseline level** $\alpha_m$ (market size, in log-sales)
- **Linear trend** $\beta_m \cdot t / 52$ (yearly drift, market-specific)
- **Seasonality** $A_m \cdot \sin(2 \pi t / 52 + \phi_m)$ (annual)
- **Common shock** $s_t$ (concurrent across markets — economic conditions, holidays)
- **Idiosyncratic noise** $\varepsilon_{m,t}$

In **treatment phase** ($t \ge 52$), 10 markets receive an additive lift $\theta_m$ on log-sales (multiplicative on raw scale).  Treated indices are drawn at random; the planted lifts are $\theta_m \sim \mathcal{N}(0.05, 0.02^2)$ — average 5 % uplift on raw sales, with non-trivial cross-market heterogeneity.

The **shared common shock** $s_t$ is what forces honest causal methods (DiD, synthetic control) over a naive pre/post — without it, every market goes up after week 52 by some constant and DiD coincidentally works.  We make this real-world-realistic.
""")

co("simulate", r"""
@dataclass
class GeoSim:
    n_markets: int = 30
    n_weeks: int = 104
    n_pre: int = 52
    n_treated: int = 10
    seed: int = RNG_SEED

    sales: np.ndarray = None
    treated_mask: np.ndarray = None
    treatment_start: int = None
    true_theta: np.ndarray = None


def simulate_geo(cfg: GeoSim) -> GeoSim:
    rng = np.random.default_rng(cfg.seed)
    M, T = cfg.n_markets, cfg.n_weeks
    cfg.treatment_start = cfg.n_pre

    alpha = rng.normal(10.0, 0.6, M)
    beta = rng.normal(0.04, 0.02, M)
    amp = np.abs(rng.normal(0.10, 0.03, M))
    phi = rng.uniform(0, 2 * np.pi, M)

    common_shock = np.cumsum(rng.normal(0, 0.015, T)) - 0.5 * np.sin(2 * np.pi * np.arange(T) / 52 + 0.6) * 0.04

    t = np.arange(T)
    log_sales = np.empty((M, T))
    for m in range(M):
        log_sales[m] = (
            alpha[m]
            + beta[m] * (t / 52)
            + amp[m] * np.sin(2 * np.pi * t / 52 + phi[m])
            + common_shock
            + rng.normal(0, 0.04, T)
        )

    treated_idx = rng.choice(M, size=cfg.n_treated, replace=False)
    treated_mask = np.zeros(M, dtype=bool); treated_mask[treated_idx] = True

    true_theta = np.zeros(M)
    true_theta[treated_idx] = rng.normal(0.05, 0.02, cfg.n_treated)
    log_sales[treated_idx, cfg.treatment_start:] += true_theta[treated_idx, None]

    cfg.sales = np.exp(log_sales)
    cfg.treated_mask = treated_mask
    cfg.true_theta = true_theta
    return cfg


SIM = simulate_geo(GeoSim())
sales_df = pd.DataFrame(SIM.sales.T, columns=[f"M{m:02d}" for m in range(SIM.n_markets)])
sales_df.index.name = "week"

print(f"Markets        : {SIM.n_markets}")
print(f"Weeks          : {SIM.n_weeks}  (pre {SIM.n_pre}, post {SIM.n_weeks - SIM.n_pre})")
print(f"Treated        : {int(SIM.treated_mask.sum())}  ({np.where(SIM.treated_mask)[0].tolist()})")
print(f"True theta (treated only):")
for m in np.where(SIM.treated_mask)[0]:
    print(f"  M{m:02d}  theta = {SIM.true_theta[m]:+.4f}")
print(f"  global mean  = {SIM.true_theta[SIM.treated_mask].mean():+.4f}")
print(f"  std across treated = {SIM.true_theta[SIM.treated_mask].std():.4f}")
""")

# ========================================================================
# 2. EDA
# ========================================================================
md("eda_intro", r"""
## 2. Per-market sales paths

The goal of this plot is a single visual: **before week 52 the markets move in concert** (same trend, same seasonality, common shock); **after week 52 the treated markets diverge upward** by an amount we have not made obvious.
""")

co("eda_plot", r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)

ts = np.arange(SIM.n_weeks)
for m in range(SIM.n_markets):
    if SIM.treated_mask[m]:
        axes[0].plot(ts, SIM.sales[m] / SIM.sales[m, :SIM.n_pre].mean(),
                     color="#d62728", alpha=0.7, lw=0.8)
    else:
        axes[0].plot(ts, SIM.sales[m] / SIM.sales[m, :SIM.n_pre].mean(),
                     color="#888888", alpha=0.4, lw=0.6)
axes[0].axvline(SIM.n_pre - 0.5, color="black", ls="--", lw=0.8)
axes[0].set_title("All markets (treated red, control grey)\nNormalised by pre-period mean")
axes[0].set_xlabel("week")
axes[0].set_ylabel("sales / pre-period mean")

mean_treated = SIM.sales[SIM.treated_mask].mean(axis=0)
mean_control = SIM.sales[~SIM.treated_mask].mean(axis=0)
axes[1].plot(ts, mean_treated / mean_treated[:SIM.n_pre].mean(), color="#d62728", lw=2, label="treated mean")
axes[1].plot(ts, mean_control / mean_control[:SIM.n_pre].mean(), color="#1f77b4", lw=2, label="control mean")
axes[1].axvline(SIM.n_pre - 0.5, color="black", ls="--", lw=0.8)
axes[1].set_title("Treated vs control mean trajectories (normalised)")
axes[1].set_xlabel("week"); axes[1].legend()
plt.tight_layout(); plt.show()
""")

co("balance_check", r"""
pre_levels = SIM.sales[:, :SIM.n_pre].mean(axis=1)
balance_df = pd.DataFrame({
    "market": [f"M{m:02d}" for m in range(SIM.n_markets)],
    "treated": SIM.treated_mask.astype(int),
    "pre_mean_sales": pre_levels.round(0),
    "pre_std_sales": SIM.sales[:, :SIM.n_pre].std(axis=1).round(0),
})
treated_mean = balance_df[balance_df["treated"] == 1]["pre_mean_sales"].mean()
control_mean = balance_df[balance_df["treated"] == 0]["pre_mean_sales"].mean()
t_stat, p_val = stats.ttest_ind(balance_df[balance_df["treated"] == 1]["pre_mean_sales"],
                                  balance_df[balance_df["treated"] == 0]["pre_mean_sales"])
print(f"Pre-period mean sales: treated {treated_mean:,.0f}  vs  control {control_mean:,.0f}")
print(f"t-test p-value (random assignment check): {p_val:.3f}  (>= 0.10 means good balance)")
""")

# ========================================================================
# 3. NAIVE / DiD
# ========================================================================
md("naive_intro", r"""
## 3. Two estimators that get it wrong (or close to it)

The naive **pre / post difference** on each treated market is:
$$\hat\theta_m^{\text{naive}} = \log \overline{S}_m^{\text{post}} - \log \overline{S}_m^{\text{pre}}$$

This is biased upward by *all* the post-period drift (trend, seasonality, common shock) — the entire reason geo experiments have a control group at all.

**Difference-in-Differences (DiD)** subtracts the same drift estimated on the *control markets*:
$$\hat\theta_m^{\text{DiD}} = (\log \overline{S}_m^{\text{post}} - \log \overline{S}_m^{\text{pre}}) - (\log \overline{S}_C^{\text{post}} - \log \overline{S}_C^{\text{pre}})$$

This is unbiased *if* parallel-trends holds — i.e. the treated and control mean would have moved in lockstep without the intervention.  We score both against truth.
""")

co("naive_did", r"""
def naive_estimate(market_idx: int, sales: np.ndarray, pre_end: int) -> float:
    pre_log = np.log(sales[market_idx, :pre_end]).mean()
    post_log = np.log(sales[market_idx, pre_end:]).mean()
    return post_log - pre_log


def did_estimate(market_idx: int, sales: np.ndarray, control_idx: np.ndarray, pre_end: int) -> tuple[float, float]:
    pre_log_t = np.log(sales[market_idx, :pre_end])
    post_log_t = np.log(sales[market_idx, pre_end:])
    pre_log_c = np.log(sales[control_idx, :pre_end]).mean(axis=0)
    post_log_c = np.log(sales[control_idx, pre_end:]).mean(axis=0)
    treated_diff = post_log_t.mean() - pre_log_t.mean()
    control_diff = post_log_c.mean() - pre_log_c.mean()
    se_t_pre = np.std(pre_log_t, ddof=1) / np.sqrt(pre_end)
    se_t_post = np.std(post_log_t, ddof=1) / np.sqrt(len(post_log_t))
    se_c_pre = np.std(pre_log_c, ddof=1) / np.sqrt(pre_end)
    se_c_post = np.std(post_log_c, ddof=1) / np.sqrt(len(post_log_c))
    se = np.sqrt(se_t_pre**2 + se_t_post**2 + se_c_pre**2 + se_c_post**2)
    return float(treated_diff - control_diff), float(se)


control_idx = np.where(~SIM.treated_mask)[0]
treated_idx_arr = np.where(SIM.treated_mask)[0]

naive_rows, did_rows = [], []
for m in treated_idx_arr:
    n = naive_estimate(m, SIM.sales, SIM.n_pre)
    d, d_se = did_estimate(m, SIM.sales, control_idx, SIM.n_pre)
    naive_rows.append({"market": f"M{m:02d}", "true": SIM.true_theta[m],
                       "naive_theta": n, "naive_bias": n - SIM.true_theta[m]})
    did_rows.append({"market": f"M{m:02d}", "true": SIM.true_theta[m],
                     "did_theta": d, "did_se": d_se,
                     "did_lo": d - 1.96 * d_se, "did_hi": d + 1.96 * d_se,
                     "did_bias": d - SIM.true_theta[m]})

naive_df = pd.DataFrame(naive_rows).round(4)
did_df = pd.DataFrame(did_rows).round(4)
print("Naive pre/post estimates per treated market:")
print(naive_df.to_string(index=False))
print()
print("DiD estimates per treated market:")
print(did_df.to_string(index=False))
print()
print(f"Naive mean bias    : {naive_df['naive_bias'].mean():+.4f}  RMSE {np.sqrt((naive_df['naive_bias']**2).mean()):.4f}")
print(f"DiD   mean bias    : {did_df['did_bias'].mean():+.4f}  RMSE {np.sqrt((did_df['did_bias']**2).mean()):.4f}")
""")

# ========================================================================
# 4. SYNTHETIC CONTROL
# ========================================================================
md("sc_intro", r"""
## 4. Synthetic Control (Abadie 2003)

For each treated market $m$, find a non-negative weighted combination $w$ of the **control markets** that best matches the treated market on the **pre-period** trajectory:

$$\hat w_m = \arg\min_{w \in \Delta^{|C|}} \big\| y_m^{\text{pre}} - W_C^{\text{pre}} w \big\|_2^2$$

where $\Delta^{|C|}$ is the simplex (weights sum to 1, all non-negative — forces *interpolation*, not extrapolation).  The treatment effect is the post-period difference between observed treated sales and the synthetic counterfactual:

$$\hat\theta_m = \log \overline{y}_m^{\text{post}} - \log (\overline{W_C^{\text{post}} w})$$

We hand-roll the constrained least-squares with `scipy.optimize.minimize` (SLSQP).  Bootstrap CIs come from resampling the **pre-period residuals** and re-running the fit (placebo permutations on control units would be the alternative; we use residual bootstrap for speed).
""")

co("sc_solver", r"""
def fit_synthetic_control(y_treated: np.ndarray, Y_donors: np.ndarray) -> np.ndarray:
    K = Y_donors.shape[0]
    w0 = np.ones(K) / K
    def loss(w):
        return float(np.sum((y_treated - w @ Y_donors) ** 2))
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * K
    res = minimize(loss, w0, method="SLSQP", constraints=cons, bounds=bounds,
                    options={"maxiter": 200, "ftol": 1e-9})
    return res.x


def sc_estimate(market_idx: int, sales: np.ndarray, control_idx: np.ndarray,
                pre_end: int, n_boot: int = 200) -> dict:
    y = np.log(sales[market_idx])
    Y_d = np.log(sales[control_idx])
    w = fit_synthetic_control(y[:pre_end], Y_d[:, :pre_end])
    counter_pre = w @ Y_d[:, :pre_end]
    counter_post = w @ Y_d[:, pre_end:]
    actual_post = y[pre_end:]
    point = float(actual_post.mean() - counter_post.mean())

    pre_resid = y[:pre_end] - counter_pre
    rng = np.random.default_rng(RNG_SEED + market_idx)
    boots = []
    n_post = len(actual_post)
    for _ in range(n_boot):
        sample_resid = rng.choice(pre_resid, size=n_post, replace=True)
        boots.append(point - sample_resid.mean() + np.mean(rng.choice(pre_resid, size=n_post, replace=True)))
    se = float(np.std(boots, ddof=1))
    return {"market_idx": market_idx, "weights": w, "theta_hat": point, "se": se,
            "ci_lo": point - 1.96 * se, "ci_hi": point + 1.96 * se,
            "counter_pre": counter_pre, "counter_post": counter_post, "actual_post": actual_post}


sc_results = {}
for m in treated_idx_arr:
    sc_results[m] = sc_estimate(m, SIM.sales, control_idx, SIM.n_pre, n_boot=200)

sc_rows = [{"market": f"M{m:02d}", "true": SIM.true_theta[m],
             "sc_theta": sc_results[m]["theta_hat"], "sc_se": sc_results[m]["se"],
             "sc_lo": sc_results[m]["ci_lo"], "sc_hi": sc_results[m]["ci_hi"],
             "sc_bias": sc_results[m]["theta_hat"] - SIM.true_theta[m]}
            for m in treated_idx_arr]
sc_df = pd.DataFrame(sc_rows).round(4)
print("Synthetic-Control estimates per treated market:")
print(sc_df.to_string(index=False))
print()
print(f"SC    mean bias    : {sc_df['sc_bias'].mean():+.4f}  RMSE {np.sqrt((sc_df['sc_bias']**2).mean()):.4f}")
sc_coverage = float(((sc_df['sc_lo'] <= sc_df['true']) & (sc_df['sc_hi'] >= sc_df['true'])).mean())
print(f"SC 95% coverage    : {sc_coverage*100:.1f}%   (nominal 95.0%)")
""")

co("sc_plot_one", r"""
example = treated_idx_arr[0]
res = sc_results[example]
fig, ax = plt.subplots(figsize=(11, 4))
ts = np.arange(SIM.n_weeks)
ax.plot(ts, np.log(SIM.sales[example]), color="#d62728", lw=1.4, label=f"M{example:02d} (treated)")
ax.plot(ts[:SIM.n_pre], res["counter_pre"], color="#1f77b4", lw=1.2, ls="--", label="synthetic (pre-period fit)")
ax.plot(ts[SIM.n_pre:], res["counter_post"], color="#1f77b4", lw=1.2, ls=":", label="synthetic (post counterfactual)")
ax.axvline(SIM.n_pre - 0.5, color="black", ls="--", lw=0.7)
ax.fill_between(ts[SIM.n_pre:], res["counter_post"], np.log(SIM.sales[example, SIM.n_pre:]),
                color="#d62728", alpha=0.2, label=f"observed - counter = {res['theta_hat']:+.4f}")
ax.set_title(f"Synthetic Control for M{example:02d}  (true theta {SIM.true_theta[example]:+.4f}, "
              f"estimate {res['theta_hat']:+.4f}, 95% CI [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}])")
ax.set_xlabel("week"); ax.set_ylabel("log sales")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.show()

n_nonzero_weights = int((res["weights"] > 0.01).sum())
top_donors = np.argsort(-res["weights"])[:5]
print(f"Donor weights (top 5 nonzero of {n_nonzero_weights}):")
for i, di in enumerate(top_donors):
    if res["weights"][di] > 0.01:
        market_id = control_idx[di]
        print(f"  M{market_id:02d}  weight {res['weights'][di]:.3f}")
""")

# ========================================================================
# 5. HIERARCHICAL BAYES
# ========================================================================
md("hb_intro", r"""
## 5. Hierarchical Bayesian random-effects meta-analysis

Per-market synthetic-control estimates $\hat\theta_m$ with standard errors $\sigma_m$ feed into the standard meta-analytic random-effects model:

$$\theta_m \sim \mathcal{N}(\mu, \tau^2), \qquad \hat\theta_m \sim \mathcal{N}(\theta_m, \sigma_m^2)$$

with priors

$$\mu \sim \mathcal{N}(0, 0.2^2), \qquad \tau \sim \text{Half-Cauchy}(0.05).$$

We sample the joint posterior $p(\mu, \tau, \theta_1, \ldots, \theta_M \mid \hat\theta, \sigma)$ via **hand-rolled adaptive Metropolis-Hastings**: random-walk proposal on $(\mu, \log \tau)$, then conditional Gaussian update on $\theta_m$ in closed form (because the likelihood and prior are conjugate Gaussians given $\mu, \tau$).  Two chains, 6,000 iterations each, 2,000 warmup, R-hat / ESS computed.

The output is **shrunk per-market estimates** $\theta_m^{\text{post}}$ — pulled toward the global mean by an amount proportional to $\sigma_m^2 / (\sigma_m^2 + \tau^2)$ — *plus* a posterior on the meta-analytic $(\mu, \tau)$.  The shrinkage is the "free lift" hierarchical Bayes gives you over per-market frequentist CIs.
""")

co("mcmc_kernel", r"""
def half_cauchy_logpdf(x: float, scale: float) -> float:
    if x <= 0:
        return -np.inf
    return float(np.log(2 / np.pi) - np.log(scale) - np.log(1 + (x / scale) ** 2))


def normal_logpdf(x: float, mu: float, sd: float) -> float:
    return float(-0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((x - mu) / sd) ** 2)


def run_random_effects_mcmc(theta_hat: np.ndarray, sigma: np.ndarray,
                              n_iter: int = 6000, warmup: int = 2000,
                              chain_seed: int = RNG_SEED) -> dict:
    rng = np.random.default_rng(chain_seed)
    M = len(theta_hat)

    mu = float(theta_hat.mean())
    log_tau = float(np.log(max(theta_hat.std(ddof=1), 1e-3)))
    theta = theta_hat.copy()

    samples_mu = np.empty(n_iter)
    samples_tau = np.empty(n_iter)
    samples_theta = np.empty((n_iter, M))
    samples_logp = np.empty(n_iter)

    accepts = 0
    proposal_sd = np.array([0.02, 0.3])

    def log_post(mu_, log_tau_, theta_):
        tau_ = math.exp(log_tau_)
        lp = normal_logpdf(mu_, 0.0, 0.2) + half_cauchy_logpdf(tau_, 0.05) + log_tau_
        for i in range(M):
            lp += normal_logpdf(theta_[i], mu_, tau_) + normal_logpdf(theta_hat[i], theta_[i], sigma[i])
        return lp

    cur_lp = log_post(mu, log_tau, theta)

    for it in range(n_iter):
        if it == 1500:
            proposal_sd = np.array([float(np.std(samples_mu[:1500])) * 1.5,
                                       float(np.std(np.log(np.maximum(samples_tau[:1500], 1e-6)))) * 1.5])
            proposal_sd = np.maximum(proposal_sd, 1e-3)

        mu_new = mu + rng.normal(0, proposal_sd[0])
        log_tau_new = log_tau + rng.normal(0, proposal_sd[1])
        new_lp = log_post(mu_new, log_tau_new, theta)
        if math.log(rng.uniform()) < new_lp - cur_lp:
            mu, log_tau, cur_lp = mu_new, log_tau_new, new_lp
            accepts += 1

        tau = math.exp(log_tau)
        prec_prior = 1.0 / (tau * tau)
        prec_lik = 1.0 / (sigma * sigma)
        post_var = 1.0 / (prec_prior + prec_lik)
        post_mean = post_var * (mu * prec_prior + theta_hat * prec_lik)
        theta = rng.normal(post_mean, np.sqrt(post_var))
        cur_lp = log_post(mu, log_tau, theta)

        samples_mu[it] = mu
        samples_tau[it] = math.exp(log_tau)
        samples_theta[it] = theta
        samples_logp[it] = cur_lp

    return {"mu": samples_mu[warmup:], "tau": samples_tau[warmup:],
            "theta": samples_theta[warmup:], "logp": samples_logp[warmup:],
            "accept_rate_mu_tau": accepts / n_iter}


theta_hat_arr = np.array([sc_results[m]["theta_hat"] for m in treated_idx_arr])
sigma_arr = np.array([sc_results[m]["se"] for m in treated_idx_arr])

t0 = time.time()
chain1 = run_random_effects_mcmc(theta_hat_arr, sigma_arr, n_iter=6000, warmup=2000, chain_seed=RNG_SEED)
chain2 = run_random_effects_mcmc(theta_hat_arr, sigma_arr, n_iter=6000, warmup=2000, chain_seed=RNG_SEED + 17)
print(f"MCMC: 2 chains x 6000 iter (warmup 2000) in {time.time() - t0:.1f}s")
print(f"  chain1 accept rate (mu, log tau RW): {chain1['accept_rate_mu_tau']:.3f}")
print(f"  chain2 accept rate (mu, log tau RW): {chain2['accept_rate_mu_tau']:.3f}")
""")

co("mcmc_diag", r"""
def r_hat(chains: list[np.ndarray]) -> float:
    n = chains[0].shape[0]
    M_chains = len(chains)
    chain_means = np.array([c.mean() for c in chains])
    grand_mean = chain_means.mean()
    B = n / (M_chains - 1) * np.sum((chain_means - grand_mean) ** 2)
    W = np.mean([c.var(ddof=1) for c in chains])
    var_hat = (1 - 1/n) * W + B / n
    return float(np.sqrt(var_hat / W))


def ess_simple(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return float(n)
    x_demean = x - x.mean()
    var0 = np.var(x_demean, ddof=0)
    if var0 == 0:
        return float(n)
    rho_sum = 0.0
    for k in range(1, min(200, n - 1)):
        cov = np.mean(x_demean[:-k] * x_demean[k:])
        rho_k = cov / var0
        if rho_k < 0.05:
            break
        rho_sum += rho_k
    return float(n / (1 + 2 * rho_sum))


rhat_mu = r_hat([chain1["mu"], chain2["mu"]])
rhat_tau = r_hat([chain1["tau"], chain2["tau"]])
ess_mu = ess_simple(np.concatenate([chain1["mu"], chain2["mu"]]))
ess_tau = ess_simple(np.concatenate([chain1["tau"], chain2["tau"]]))
print(f"R-hat   mu  {rhat_mu:.4f}   tau  {rhat_tau:.4f}   (target < 1.05)")
print(f"ESS     mu  {ess_mu:>6.0f}   tau  {ess_tau:>6.0f}   (target > 400)")

mu_combined = np.concatenate([chain1["mu"], chain2["mu"]])
tau_combined = np.concatenate([chain1["tau"], chain2["tau"]])
theta_combined = np.concatenate([chain1["theta"], chain2["theta"]], axis=0)

print()
print(f"Posterior mu  : mean {mu_combined.mean():+.4f}  95% CI [{np.percentile(mu_combined, 2.5):+.4f}, {np.percentile(mu_combined, 97.5):+.4f}]")
print(f"Posterior tau : mean {tau_combined.mean():.4f}  95% CI [{np.percentile(tau_combined, 2.5):.4f}, {np.percentile(tau_combined, 97.5):.4f}]")
print(f"True mu       : {SIM.true_theta[SIM.treated_mask].mean():+.4f}")
print(f"True tau      : {SIM.true_theta[SIM.treated_mask].std(ddof=1):.4f}")
""")

co("mcmc_traceplot", r"""
fig, axes = plt.subplots(2, 2, figsize=(12, 5.5))
axes[0, 0].plot(chain1["mu"], color="#1f77b4", lw=0.5, alpha=0.8, label="chain 1")
axes[0, 0].plot(chain2["mu"], color="#ff7f0e", lw=0.5, alpha=0.8, label="chain 2")
axes[0, 0].set_title("trace: mu (global mean)"); axes[0, 0].legend(fontsize=8)

axes[0, 1].hist(np.concatenate([chain1["mu"], chain2["mu"]]), bins=60, color="#1f77b4", alpha=0.7)
axes[0, 1].axvline(SIM.true_theta[SIM.treated_mask].mean(), color="red", ls="--", lw=1.5, label="true mu")
axes[0, 1].set_title("posterior: mu"); axes[0, 1].legend(fontsize=8)

axes[1, 0].plot(chain1["tau"], color="#1f77b4", lw=0.5, alpha=0.8)
axes[1, 0].plot(chain2["tau"], color="#ff7f0e", lw=0.5, alpha=0.8)
axes[1, 0].set_title("trace: tau (heterogeneity)")

axes[1, 1].hist(np.concatenate([chain1["tau"], chain2["tau"]]), bins=60, color="#2ca02c", alpha=0.7)
axes[1, 1].axvline(SIM.true_theta[SIM.treated_mask].std(ddof=1), color="red", ls="--", lw=1.5, label="true tau")
axes[1, 1].set_title("posterior: tau"); axes[1, 1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# ========================================================================
# 6. FOREST PLOT + COVERAGE
# ========================================================================
md("forest_intro", r"""
## 6. Forest plot, coverage, and shrinkage

The forest plot is the operational deliverable: per-market frequentist SC estimates with their 95 % CIs, alongside the **shrunken Bayesian posteriors**.  The shrinkage moves each market toward the global mean by an amount that depends on the relative size of $\sigma_m$ and $\tau$ — markets with noisy SC fits get pulled more, markets with tight SC fits get pulled less.  The global $\mu$ posterior sits at the bottom as the meta-analytic answer.

We score **coverage** at the 95 % level for each estimator: how often does the CI contain the truth?  An honest CI hits 95 %.  A miscalibrated one over- or under-shoots.
""")

co("forest_plot", r"""
hb_summary = []
for j, m in enumerate(treated_idx_arr):
    samples = theta_combined[:, j]
    hb_summary.append({"market": f"M{m:02d}",
                        "true": SIM.true_theta[m],
                        "sc_theta": sc_results[m]["theta_hat"],
                        "sc_lo": sc_results[m]["ci_lo"],
                        "sc_hi": sc_results[m]["ci_hi"],
                        "hb_theta": float(samples.mean()),
                        "hb_lo": float(np.percentile(samples, 2.5)),
                        "hb_hi": float(np.percentile(samples, 97.5))})
hb_df = pd.DataFrame(hb_summary).round(4)
hb_df["sc_bias"] = hb_df["sc_theta"] - hb_df["true"]
hb_df["hb_bias"] = hb_df["hb_theta"] - hb_df["true"]
hb_df["sc_covers"] = ((hb_df["sc_lo"] <= hb_df["true"]) & (hb_df["sc_hi"] >= hb_df["true"])).astype(int)
hb_df["hb_covers"] = ((hb_df["hb_lo"] <= hb_df["true"]) & (hb_df["hb_hi"] >= hb_df["true"])).astype(int)
hb_df["shrinkage"] = (hb_df["hb_theta"] - hb_df["sc_theta"]) / (mu_combined.mean() - hb_df["sc_theta"]).replace(0, 1)
hb_df.round(4).set_index("market")
""")

co("forest_plot_render", r"""
fig, ax = plt.subplots(figsize=(11, 6))
y_pos = np.arange(len(hb_df))[::-1]
ax.errorbar(hb_df["sc_theta"].values, y_pos - 0.18,
             xerr=[hb_df["sc_theta"] - hb_df["sc_lo"], hb_df["sc_hi"] - hb_df["sc_theta"]],
             fmt="o", color="#1f77b4", capsize=3, label="Synthetic Control 95% CI")
ax.errorbar(hb_df["hb_theta"].values, y_pos + 0.18,
             xerr=[hb_df["hb_theta"] - hb_df["hb_lo"], hb_df["hb_hi"] - hb_df["hb_theta"]],
             fmt="s", color="#d62728", capsize=3, label="Hierarchical Bayes 95% CrI")
ax.scatter(hb_df["true"].values, y_pos, marker="x", color="black", s=80, label="ground truth", zorder=5)

ax.axvline(mu_combined.mean(), color="green", ls="--", lw=1, label=f"posterior mean(mu)={mu_combined.mean():+.4f}")
ax.fill_betweenx([min(y_pos) - 1, max(y_pos) + 1],
                   np.percentile(mu_combined, 2.5), np.percentile(mu_combined, 97.5),
                   color="green", alpha=0.12, label="95% CrI(mu)")

ax.set_yticks(y_pos); ax.set_yticklabels(hb_df["market"].values)
ax.set_xlabel("treatment effect on log-sales (theta)")
ax.set_title("Forest plot: per-market SC vs hierarchical-Bayes shrunken estimates")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(min(y_pos) - 1, max(y_pos) + 1)
plt.tight_layout(); plt.show()
""")

co("coverage_table", r"""
estimator_summary = pd.DataFrame({
    "estimator": ["naive", "DiD", "synthetic_control", "hierarchical_bayes"],
    "mean_bias": [
        naive_df["naive_bias"].mean(),
        did_df["did_bias"].mean(),
        sc_df["sc_bias"].mean(),
        hb_df["hb_bias"].mean(),
    ],
    "RMSE": [
        np.sqrt((naive_df["naive_bias"] ** 2).mean()),
        np.sqrt((did_df["did_bias"] ** 2).mean()),
        np.sqrt((sc_df["sc_bias"] ** 2).mean()),
        np.sqrt((hb_df["hb_bias"] ** 2).mean()),
    ],
    "coverage_95": [
        np.nan,
        float(((did_df["did_lo"] <= did_df["true"]) & (did_df["did_hi"] >= did_df["true"])).mean()),
        float(((sc_df["sc_lo"] <= sc_df["true"]) & (sc_df["sc_hi"] >= sc_df["true"])).mean()),
        float(hb_df["hb_covers"].mean()),
    ],
}).round(4).set_index("estimator")
estimator_summary
""")

# ========================================================================
# 7. PPC
# ========================================================================
md("ppc_intro", r"""
## 7. Posterior predictive checks

Standard hierarchical-Bayes diagnostic: simulate fake $\hat\theta_m^{\text{rep}}$ from the posterior, compare to the observed $\hat\theta_m$.  If the model is well-specified, the observed values fall inside the simulated distribution.  Two views:

- **Per-market PPC band** — observed $\hat\theta_m$ should lie inside its posterior-predictive 95 % interval.
- **Global Bayesian p-value** — Pr(min(rep) <= min(obs)), Pr(max(rep) >= max(obs)).  Values near 0 or 1 indicate model misspecification on the tails.
""")

co("ppc", r"""
def ppc_replicates(theta_samples, mu_samples, tau_samples, sigma_arr, n_replicate: int = 1000):
    rng = np.random.default_rng(RNG_SEED + 42)
    M = sigma_arr.shape[0]
    n_post = theta_samples.shape[0]
    sample_idx = rng.integers(0, n_post, size=n_replicate)
    rep = np.empty((n_replicate, M))
    for r, i in enumerate(sample_idx):
        rep[r] = rng.normal(theta_samples[i], sigma_arr)
    return rep


rep = ppc_replicates(theta_combined, mu_combined, tau_combined, sigma_arr, n_replicate=2000)
ppc_lo = np.percentile(rep, 2.5, axis=0)
ppc_hi = np.percentile(rep, 97.5, axis=0)
ppc_in = ((theta_hat_arr >= ppc_lo) & (theta_hat_arr <= ppc_hi))

bayes_p_min = float(np.mean(rep.min(axis=1) <= theta_hat_arr.min()))
bayes_p_max = float(np.mean(rep.max(axis=1) >= theta_hat_arr.max()))

print(f"PPC coverage of observed (per-market 95% PPC interval): {ppc_in.mean()*100:.1f}%")
print(f"Bayesian p-value (min stat): {bayes_p_min:.3f}   (extreme = misfit)")
print(f"Bayesian p-value (max stat): {bayes_p_max:.3f}")

fig, ax = plt.subplots(figsize=(10, 4))
y_pos = np.arange(len(treated_idx_arr))[::-1]
for i, m in enumerate(treated_idx_arr):
    ax.scatter(rep[:200, i], np.full(200, y_pos[i]), s=4, color="#1f77b4", alpha=0.15)
ax.scatter(theta_hat_arr, y_pos, marker="x", color="red", s=80, label="observed theta_hat", zorder=5)
ax.set_yticks(y_pos); ax.set_yticklabels([f"M{m:02d}" for m in treated_idx_arr])
ax.set_xlabel("theta")
ax.set_title("Posterior predictive check: simulated theta_hat (blue dots) vs observed (red x)")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.show()
""")

# ========================================================================
# 8. POWER / MDE
# ========================================================================
md("power_intro", r"""
## 8. Power / minimum detectable effect

For next-cycle planning the team needs to know: **how big a true lift can we reliably detect** with the current geo design (10 treated × 52 weeks)?  We compute the **MDE at 80 % power** by simulating from the data-generating process at varying true-lift levels and counting rejections of $H_0$: $\theta = 0$ via the synthetic-control bootstrap CI.
""")

co("power", r"""
def power_at_lift(true_lift: float, n_sims: int = 50, seed: int = RNG_SEED):
    rejections = []
    for s in range(n_sims):
        cfg = GeoSim(seed=seed + s)
        cfg = simulate_geo(cfg)
        cfg.true_theta[cfg.treated_mask] = true_lift
        log_sales = np.log(cfg.sales)
        log_sales[cfg.treated_mask, cfg.treatment_start:] -= np.array([t for t in cfg.true_theta[cfg.treated_mask]])[:, None]
        log_sales[cfg.treated_mask, cfg.treatment_start:] += true_lift
        cfg.sales = np.exp(log_sales)
        rejs = []
        ctrl = np.where(~cfg.treated_mask)[0]
        for m in np.where(cfg.treated_mask)[0]:
            r = sc_estimate(m, cfg.sales, ctrl, cfg.treatment_start, n_boot=80)
            rejs.append((r["ci_lo"] > 0) | (r["ci_hi"] < 0))
        rejections.append(np.mean(rejs))
    return float(np.mean(rejections))


t0 = time.time()
lift_grid = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
power_curve = [(lift, power_at_lift(lift, n_sims=12)) for lift in lift_grid]
print(f"Power curve computed in {time.time() - t0:.1f}s")
power_df = pd.DataFrame(power_curve, columns=["true_lift", "power_at_80pct_test_5pct"])
power_df.round(3)
""")

co("power_plot", r"""
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(power_df["true_lift"], power_df["power_at_80pct_test_5pct"], "o-", color="#1f77b4", lw=1.5, ms=7)
ax.axhline(0.8, color="red", ls="--", lw=1, label="80% power target")
ax.axhline(0.05, color="grey", ls="--", lw=1, label="5% type-I error baseline")
ax.set_xlabel("true lift theta")
ax.set_ylabel("rejection rate")
ax.set_title("Power curve: synthetic-control rejection rate vs true treatment lift")
ax.legend()
plt.tight_layout(); plt.show()
""")

# ========================================================================
# 9. DECISION MEMO
# ========================================================================
md("decision_memo", r"""
## 9. Decision memo

**Recommendation.**  Report **the hierarchical-Bayes shrunken estimate $\theta_m^{\text{post}}$** as the per-market lift, alongside the SC point estimate as a sanity check.  Use the **global $\mu$ posterior** as the meta-analytic input to next quarter's plan.  Use the **MDE curve** to set the budget cap on small-effect markets where the design has insufficient power.

**Why hierarchical Bayes rather than just synthetic control?**

- The SC-only forest plot gives ten **independent** estimates and ignores the fact that they all measure variants of the same underlying campaign — the team is implicitly going to *average them in their head* anyway.  Doing the averaging explicitly via a partial-pooling model is more honest *and* tighter.
- Per-market SC CIs are wide because pre-period residual bootstrap is high-variance.  HB **shrinks** noisy markets toward the global mean and tightens the CIs in proportion to how much information the cross-market data add — exactly what shrinkage is for.
- HB returns a **posterior on $\tau$** — the *heterogeneity* across markets — which is itself a deliverable.  Large $\tau$ means market-specific creative is worth funding; small $\tau$ means the campaign's effect is portable and the team should focus on rollout speed rather than per-market customisation.

**Why not skip SC and go straight to a fully-Bayesian causal model?**

- A fully-Bayesian state-space model per market (Brodersen et al. 2015 / TBR) is the right choice when you have **few markets and rich pre-period covariate paths**.  It's heavier to fit and per-market inference dominates the runtime budget.
- The **two-stage** approach used here (point-estimate-with-SE per market → meta-analysis) is the standard random-effects framework, requires only Gaussian conditionals, and is fast enough to refit at each decision point.  It is also straightforward to extend: replace the per-market SC fit with a Bayesian state-space fit when you have the budget.

**What I would do next.**

1. **Replace residual bootstrap with placebo-permutation CIs** (Abadie's original suggestion: assign treatment to each control unit, refit SC, build a null distribution).  More principled when the number of controls is small.
2. **Spike-and-slab priors on $\theta_m$** — for each market, the prior is a mixture of "no effect" and "lift drawn from $\mathcal{N}(\mu, \tau^2)$".  Posterior probability of "no effect" is the data-driven answer to "should we even keep running this campaign in M07?".
3. **Hierarchical structure on $\mu$ itself** — region-level pooling: north-east markets share a $\mu_{\text{NE}}$ that's pooled toward a national $\mu$.  Three-level hierarchies are a natural extension and help when sample size per level is small.
""")

# ========================================================================
# 10. PRODUCTION HYGIENE
# ========================================================================
md("hygiene_intro", r"""
## 10. Production hygiene

Persist:

- **MCMC samples** for $(\mu, \tau, \theta)$ — the operational deliverable; the planner pulls quantiles directly from this.
- **Per-market SC weights** — for transparency / audit (which donor markets contribute to each treated counterfactual).
- **Posterior summary table** as JSON.
- **Model card** with limitations.
""")

co("persist", r"""
artefact_dir = DATA_DIR / "production"
artefact_dir.mkdir(exist_ok=True)

joblib.dump({"mu": mu_combined, "tau": tau_combined, "theta": theta_combined,
              "treated_idx": treated_idx_arr.tolist(),
              "true_theta": SIM.true_theta[SIM.treated_mask].tolist()},
             artefact_dir / "mcmc_samples.joblib")

joblib.dump({m: {"weights": sc_results[m]["weights"].tolist(), "theta_hat": sc_results[m]["theta_hat"],
                  "se": sc_results[m]["se"]} for m in treated_idx_arr},
             artefact_dir / "sc_per_market.joblib")

posterior_summary = {
    "mu_mean": float(mu_combined.mean()),
    "mu_ci": [float(np.percentile(mu_combined, 2.5)), float(np.percentile(mu_combined, 97.5))],
    "tau_mean": float(tau_combined.mean()),
    "tau_ci": [float(np.percentile(tau_combined, 2.5)), float(np.percentile(tau_combined, 97.5))],
    "per_market": [
        {"market": f"M{m:02d}",
         "true_theta": float(SIM.true_theta[m]),
         "sc_theta": float(sc_results[m]["theta_hat"]),
         "sc_se": float(sc_results[m]["se"]),
         "hb_theta_mean": float(theta_combined[:, j].mean()),
         "hb_theta_ci": [float(np.percentile(theta_combined[:, j], 2.5)),
                          float(np.percentile(theta_combined[:, j], 97.5))]}
        for j, m in enumerate(treated_idx_arr)
    ],
    "rhat": {"mu": rhat_mu, "tau": rhat_tau},
    "ess":  {"mu": ess_mu,  "tau": ess_tau},
}
(artefact_dir / "posterior_summary.json").write_text(json.dumps(posterior_summary, indent=2))

print("Persisted artefacts:")
for p in sorted(artefact_dir.glob("*")):
    print(f"  {p.name:<32s}  {p.stat().st_size/1024:>8.1f} KB")
""")

co("model_card", r"""
def make_card() -> dict:
    return {
        "name": "geo_experiments_hierarchical_bayesian",
        "version": "1.0.0",
        "task": "geo-experiment causal estimation with random-effects partial pooling",
        "data": {
            "source": "synthetic",
            "n_markets": int(SIM.n_markets), "n_weeks": int(SIM.n_weeks),
            "n_pre_weeks": int(SIM.n_pre), "n_treated": int(SIM.treated_mask.sum()),
            "true_global_mean": float(SIM.true_theta[SIM.treated_mask].mean()),
            "true_heterogeneity": float(SIM.true_theta[SIM.treated_mask].std(ddof=1)),
        },
        "estimator_quality": estimator_summary.to_dict(orient="index"),
        "mcmc": {"chains": 2, "iter_per_chain": 6000, "warmup": 2000,
                  "rhat_mu": rhat_mu, "rhat_tau": rhat_tau,
                  "ess_mu": ess_mu, "ess_tau": ess_tau,
                  "accept_rate": float(np.mean([chain1["accept_rate_mu_tau"], chain2["accept_rate_mu_tau"]]))},
        "posterior": {"mu_mean": float(mu_combined.mean()),
                       "mu_ci": [float(np.percentile(mu_combined, 2.5)),
                                  float(np.percentile(mu_combined, 97.5))],
                       "tau_mean": float(tau_combined.mean()),
                       "tau_ci": [float(np.percentile(tau_combined, 2.5)),
                                   float(np.percentile(tau_combined, 97.5))]},
        "ppc": {"per_market_coverage_at_95": float(ppc_in.mean()),
                 "bayes_p_min": bayes_p_min, "bayes_p_max": bayes_p_max},
        "power_curve": [{"true_lift": float(l), "rejection_rate": float(p)} for l, p in power_curve],
        "intended_use": "Marketing geo-experiment lift estimation with calibrated per-market and global CrIs.",
        "limitations": [
            "Synthetic data — methodology transfers but operational priors (mu prior scale, tau prior scale) should be re-elicited from historical campaign data.",
            "Two-stage estimation (SC point estimate -> meta-analysis) treats sigma_m as known and ignores its sampling variance; a fully-Bayesian state-space alternative removes this approximation at higher compute cost.",
            "Random-effects assumes exchangeability of markets; in practice markets have known structural differences (urban/rural, region) that should enter as level-2 covariates.",
            "Pre-period bootstrap residual CI is conservative when the synthetic-control fit is good and anti-conservative when it's poor; placebo permutations on control units are the principled alternative.",
        ],
    }


card = make_card()
card_path = DATA_DIR / "model_card.json"
card_path.write_text(json.dumps(card, indent=2))
print(f"Wrote model card to {card_path}")
print(json.dumps({"name": card["name"], "mu_mean_post": card["posterior"]["mu_mean"],
                  "true_mu": card["data"]["true_global_mean"]}, indent=2))
""")

# ========================================================================
# 11. LIMITATIONS
# ========================================================================
md("limitations", r"""
## 11. Limitations and next steps

**Data.**

- Synthetic.  Real geo data has covariates (CPI, weather, store openings, competitor activity) that should enter the SC weight optimisation as auxiliaries; the methodology generalises but the implementation here uses sales alone.
- 30 markets is the typical low-end of geo experiments; 100-500 markets is more common at the brand level.  At larger market counts the meta-analytic shrinkage is more powerful.

**Estimators.**

- **SC residual bootstrap CI** assumes the pre-period residual distribution is representative of post-period sampling noise; this can break when the donor pool is small or pre-period fit is bad.  Placebo permutations on control units are the more rigorous alternative.
- **DiD** assumes parallel trends; we did not test the assumption (e.g., via event-study coefficients on pre-period leads).  A real deployment would.
- **Hierarchical Bayes** assumes exchangeability of markets given the random effect.  In production, observable market characteristics (region, population, baseline-sales tier) should enter as level-2 covariates.

**Inference.**

- Hand-rolled MH with a Gibbs step on $\theta$ — works but does not scale to high-dimensional latents.  For 200 markets with covariates, NUTS via NumPyro or PyMC is the standard.
- Two chains for R-hat is the bare minimum; production deployments would run 4-8.
- ESS is computed via a simple positive-autocorrelation cutoff — the IRT (initial positive sequence) estimator from Geyer is more standard and slightly more conservative.

**Production.**

- No real-time updating: the pipeline retrains end-to-end at each decision point.  Streaming HB updates (sequential Monte Carlo, online variational inference) are the next layer.
- No spike-and-slab on $\theta_m$ — every market is implicitly assumed to have **some** effect.  In real campaigns a fraction of markets simply do not respond; a mixture prior would surface those explicitly.
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
