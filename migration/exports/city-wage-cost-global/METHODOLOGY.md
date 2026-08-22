# Methodology Notes

## Core idea

The notebooks estimate whether higher nominal income in cities is enough to
compensate for higher local costs.

## Core index

```text
real income index = nominal income index / cost index * 100
```

A city with income index 130 and cost index 150 has a real income index of 86.7.
It pays more, but the worker is worse off in purchasing-power terms.

## Data vintage

| Region | Source | Vintage |
| --- | --- | --- |
| US | BLS OEWS metropolitan (MSA file) | May 2024 |
| US | BEA Regional Price Parities (FRED `RPPALL*`, `RPPSERVERENT*`) | 2024 |
| UK | ONS Explore Local Statistics: gross median weekly pay | latest release |
| UK | ONS Explore Local Statistics: housing affordability ratio | latest release |
| UK | ONS Relative Regional Consumer Price Levels | 2016 |
| EU | Eurostat `nama_10r_2hhinc` (B6N, EUR per inhabitant) | 2023–2024 by region |
| EU | Eurostat `tec00120` price-level index (EU27 = 100) | 2025 |

## US-specific approach

OEWS publishes several files inside the metro ZIP; the analysis uses the **MSA**
file (5-digit CBSA codes), not the "BOS"/balance-of-state nonmetro file (7-digit
codes that never match a BEA/FRED RPP series). Real wage = nominal wage ÷
(all-items RPP / 100). Because RPP is all-items, the US cost adjustment already
includes housing. Occupation-level results use the same RPP per metro.

## UK-specific approach

The UK notebook uses local median weekly pay (annualised as 52 × weekly pay) and
the ONS house-price-to-earnings affordability ratio. Housing is treated
separately because it dominates local cost differences and because current
local-authority consumer-price parities are not published. The only general-price
adjustment available is the flat ONS 2016 regional benchmark (London ≈ 107, rest
≈ 98–100), so it barely moves rankings by construction — which is exactly why the
analysis foregrounds housing instead.

## EU-specific approach

The EU notebook uses NUTS2 **net disposable household income per inhabitant**
(`nama_10r_2hhinc`, `na_item = B6N`, `unit = EUR_HAB`, `direct = BAL`) deflated by
the **country-level** price-level index (`tec00120`). Two choices matter:

1. **Per inhabitant, not totals.** The same dataset also carries `MIO_EUR`
   (whole-region totals in millions). Mixing units would rank the largest regions
   first instead of the richest per person; the notebook pins a single
   `EUR_HAB` measure.
2. **NUTS2 only.** Income is restricted to 4-character NUTS2 codes so the ranking
   compares like with like rather than mixing country, NUTS1 and NUTS2 levels.

This captures **between-country** price differences well but **within-country**
city costs (above all housing) poorly. The optional manual housing-cost merge is
where that gap can be closed.

## Comparative approach

The comparative notebook indexes each region's places to that region's own median
(median = 100) on nominal and real income, then measures (a) the rank correlation
between nominal and real income and (b) how many index points the top-decile
nominal premium loses after the cost adjustment. The cost adjustment differs by
region (US all-items RPP; EU national price level; UK non-housing only), so the
comparison is conceptual, not a claim that the three numbers are identically
constructed.

## Strong claims versus weak claims

Stronger claims (supported by the executed notebooks):

- nominal income differs strongly across places, and the nominal "city premium"
  is nearly universal;
- the cost adjustment compresses that premium everywhere and reshuffles rankings
  (Spearman nominal-vs-real ≈ 0.71 US, ≈ 0.44 EU);
- some high-wage places look worse after adjustment (San Francisco below Boulder
  in real terms; Copenhagen and Dublin falling ~85+ EU rank places; Miami near
  the US real-wage bottom);
- the premium is occupation-specific in the US (high-skill keeps it; some
  low-wage service work goes negative) and housing-driven in the UK.

Weaker claims (not supported without more data):

- exact individual financial gain from migration;
- exact city-level affordability across all EU cities (national PPP misses city
  housing);
- occupation-level EU city premiums without restricted microdata;
- within-city differences (MSA/region averages hide core-vs-suburb housing gaps).
