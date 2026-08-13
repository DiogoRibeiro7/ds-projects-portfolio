# Two populations, one country

**What Portugal's 2025 GDP-per-capita revision does and does not mean.**

On 22 June 2026, Statistics Portugal (INE) published its *Estimativas de População Residente*
for 2025: **11,424,031** residents, on a new fully administrative basis, with 2021–2024 revised
by the same method. A widely shared inference followed: more people dividing the same output
means GDP per capita must fall, from a preliminary **81.3%** of the EU average to **77%**.

Testing that needs one question answered first, and it is a question of fact:

> **Which population is the published GDP-per-capita index actually divided by?**

## The finding

Portugal has two official population figures for 2025, and they differ by 5.6%:

| series | 2025 | what it is |
| --- | --- | --- |
| Eurostat `demo_gind` / INE | 11,405,627 | revised demographic estimate |
| Eurostat `nama_10_pe` (POP_NC) | **10,804,160** | **national-accounts population** |

Every published per-capita national-accounts figure divides by the second. The accounts have
not been rebenchmarked onto the revision — they could not have been, since the 2025 accounts
were compiled before INE published it in June 2026.

This is recoverable from published data without assuming anything, because Eurostat publishes
GDP per capita in euro directly:

```text
population inside the index = GDP / GDP per capita in euro
                            = 306,749.6 M€ / 28,390 €
                            = 10,804,847        ← matches nama_10_pe to 0.006%
```

**So the denominator correction is owed.** Substituting the revised resident population and
holding GDP fixed takes the 2025 index from 81.4 to **77.2**. The article's arithmetic is right
and its 1.056 correction factor is close to the true ratio between the two bases.

### But holding GDP fixed is an assumption

The two bases differ *because* the accounts have not absorbed some **601,467 residents**. At
INE's observed migrant age structure (86.1% aged 15–64) and the OECD foreign-born employment
rate (76.5%), those residents imply about **396,000 workers**. A denominator-only correction
assumes measured GDP already captures all of their output. That assumption deserves a name:

> **Statistical capture** (φ) — the share of the uncounted residents' output the national
> accounts already record.

| φ | meaning | 2025 index |
| --- | --- | --- |
| 1.0 | accounts already record all of it | **77.2** ← the article's claim |
| 0.0 | accounts recorded none of it | **81.7** |
| — | published index, on the accounts population | 81.4 |

### The data cannot measure φ

This is the least comfortable finding, and it took three attempts to reach.

National-accounts employment has grown faster than the national-accounts population base
(48.1% → 50.0% of it between 2019 and 2025), which looks like the accounts absorbing uncounted
workers. Two things kill that reading:

- **The employment count shares the blind spot.** It is grossed up on the *same*
  un-rebenchmarked population. If ~600,000 residents sit outside that benchmark, their work sits
  outside the count by construction — so the ratio says nothing about what is in the blind spot.
- **Employment rates rose across Europe.** Portugal's +4.1pp since 2019 is +1.2pp against the
  EU27, +0.4pp against Spain and **−1.1pp against Ireland** — the two best comparators being
  countries that also absorbed large migration inflows. There is no consistent excess.

An earlier version of this project reported a capture range of 0.24–0.64 derived from exactly
that excess-employment calculation without controlling for the European trend. It was an
artefact and has been withdrawn; `test_the_portugal_case_shows_no_consistent_excess` pins the
correction.

### What does bear on φ: compilation practice

**For high capture** — GDP is built mainly from production-side records (VAT turnover, corporate
filings, structural business statistics) in which a firm's output appears regardless of whether
the demographic count knows its staff exist. The population benchmark is simply not in that
measurement chain, and the accounts carry a separate exhaustiveness adjustment for undeclared
work.

**For low capture** — some components *are* population-benchmarked: imputed rents, parts of
household final consumption, non-market output. Those would be understated on a base 5.6% short.

The population-benchmarked components are a minority of GDP, so on balance capture is high and
the answer sits near **77.2**, with **81.7** as the outer bound. That is a reasoned judgement
about compilation practice, not a measurement, and the project does not dress it up as one.
When the accounts are rebenchmarked, numerator and denominator will share a basis and φ will
stop mattering.

## What the Bayesian model adds

Once the headline question is framed, the model becomes something that can be **scored**, since
2025 is now published:

| model | 2025 GDP median | actual | error | 90% interval width |
| --- | --- | --- | --- | --- |
| Full-history estimation window | 316.3 bn | 306.7 bn | +3.1% | ~44 bn |
| Euro-era window (1999–) | **302.3 bn** | 306.7 bn | **−1.4%** | ~19 bn |

That 2025 comparison is a **conditional nowcast, not a forecast**: the model is handed INE's
announced real growth and has to supply the deflator, so the miss is deflator error (2.4% drawn
against 3.9% realised). The unconditional evidence is the back-test below.

Portuguese nominal series break decisively at euro entry — nominal GDP growth averaged 15.5% a
year before 1985 against 3.7% from 1999, the deflator 12.0% against 2.6%. Fitting across that
break biases the deflator prediction to 6.4% (actual: 3.9%) and inflates its SD to 6.0pp, which
lets a weak regression outvote the direct real-growth signal.

**Honest limitations.** Restricting to the euro era cut mean absolute back-test error from 6.0%
to 3.9% and halved the interval width, but coverage *fell* from 80% to 67% against a 90% target.
Every miss is a regime shift — 2011–12 austerity, 2020 COVID, the 2022–23 inflation surge.
Outside those years the model runs at 80% coverage and 2.8% error. Loosening the inverse-gamma
prior on the residual variance by a factor of six leaves coverage completely unmoved, so the
limitation is reported rather than tuned away.

## The methodological trap

There is an inviting shortcut that silently destroys the central test, and it looks like rigour
while you are doing it: deriving the PPP conversion factor as `GDP / population / gdp_pc_pps`
using whichever population you had in mind, then "recovering" the population from the index.
That is circular — it returns your assumption, dressed as a measurement.

```text
assume 10,804,160  →  PPP 0.83937  →  "recovers" 10,804,160   ✓ matches the assumption
assume 11,405,627  →  PPP 0.79511  →  "recovers" 11,405,627   ✓ matches the assumption
```

The derivation must run through two independently published figures — GDP per capita in euro
and GDP per capita in PPS — so no population enters it at all. `derive_purchasing_power_parity`
takes exactly those two arguments and cannot be called any other way;
`test_ppp_is_independent_of_any_population_assumption` pins it.

A cross-check confirms which route is right: the non-circular PPP of 0.839 implies a price level
of 83.9, consistent with Eurostat's published price level index (82.5 in 2024). The circular
alternative implies 79.5, which is not.

## Layout

```text
data/
  known_observations.csv    official anchors + the article's own numbers, each with a source
  source_catalog.json       every API used, with vintage warnings
  raw/                      cached API responses (git-ignored; regenerated on first run)
notebooks/
  01_bayesian_gdp_population_revision.ipynb
outputs/                    generated tables (verdict, capture curve, back-test, scores)
src/pt_gdp_bayes/
  bayes.py                  conjugate Normal-Inverse-Gamma regression, normal-normal update
  data_sources.py           fetch-then-cache layer for World Bank, Eurostat and OECD
  reconciliation.py         index inversion, the two population bases, employment absorption
  model.py                  posteriors, back-test, outturn scoring, capture sensitivity
  pipeline.py               shared config and end-to-end run
tests/                      131 tests
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

python scripts/run_analysis.py            # uses data/raw/ cache; --refresh to re-download
jupyter lab notebooks/01_bayesian_gdp_population_revision.ipynb
pytest tests/ -q
```

`data/raw/` is generated on first run. The first run needs network access; every run after that
is offline.

> **Note on version control.** The repository's root `.gitignore` excludes `**/data/` and
> `outputs/`. The cache and generated outputs are regenerable and stay ignored, but
> `known_observations.csv` and `source_catalog.json` are hand-curated source — the official
> anchors and their citations — and without them the project will not run from a fresh clone,
> so both are force-added (`git add -f`) and tracked.

## Data sources

| source | role |
| --- | --- |
| INE, *Estimativas de População Residente 2025* (22 Jun 2026) | revised resident population; migrant age structure (86.1% aged 15–64); foreign-resident count |
| Eurostat `nama_10_pe` | **national-accounts population** — the denominator actually inside the published index |
| Eurostat `demo_gind` | revised demographic population — the right denominator for output per person |
| Eurostat `nama_10_pc` | GDP per capita in **euro** and in PPS — the two published figures that make the recovery non-circular |
| Eurostat `nama_10_gdp`, `tec00114` | nominal GDP and the published headline index |
| Eurostat `nama_10_a10_e` | national-accounts employment, for the capture evidence |
| World Bank API | long GDP, deflator and labour history from 1960. Its population series tracks the accounts basis, so it is never used as "the population" |
| OECD International Migration Database | employment rate by place of birth |

## The lesson

Three passes at this question went wrong before it came out right, in three different
directions, and the failures rhyme:

1. **Correcting a per-capita statistic** without establishing which denominator was already
   inside it.
2. **A circular derivation** — computing a conversion factor from an assumed population, then
   "recovering" the population from it, which returns the assumption wearing the costume of a
   measurement.
3. **Attributing a Europe-wide employment recovery** to one country's statistical system,
   producing a confident capture range that a single comparison against Spain and Ireland
   dissolved.

Each error produced a *more* precise-looking answer than the evidence supported. Precision is
the thing to distrust first.

Reconstruct a published statistic before correcting it, and check that any signal you find is
absent from the controls. If you cannot recover a statistic's inputs from published data, you do
not yet know what you are adjusting.
