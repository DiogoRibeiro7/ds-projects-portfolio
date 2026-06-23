# Portugal GDP Income Distribution Notebook

This notebook analyses Portugal's GDP by the income approach using Eurostat data via DBnomics, and places it next to actual GDP and broader living-standard indicators.

## Contents

### Functional distribution (descriptive)

- compensation of employees, gross operating surplus + mixed income, and net taxes as % of GDP;
- accounting-identity check, three-year moving averages, crisis annotations, capital-vs-wages comparison.

### Statistical analysis

- descriptive distribution and normality tests (Shapiro–Wilk, Jarque–Bera);
- stationarity / order of integration (ADF, KPSS) and autocorrelation diagnostics;
- HAC (Newey–West) trend estimation;
- structural breaks (Chow tests and a sup-F / Quandt scan);
- regime comparison (Welch t-test, Mann–Whitney U);
- capital–wage correlation and Engle–Granger cointegration.

### Comparison with actual GDP

- real GDP growth (chain-linked volumes) and the GDP level;
- cyclicality of each share (HAC regressions of share changes on real GDP growth).

### Methodological refinements

- compositional log capital-to-labour ratio;
- Zivot–Andrews unit-root test with an endogenous break;
- interrupted time-series (segmented regression) quantifying the 2012 break;
- persistence-aware AR(1) bootstrap of the break's significance;
- euro-area benchmark.

### Living standards and inequality

- real GDP per capita (and PPS), population, life expectancy, unemployment;
- inequality and poverty (Gini, S80/S20, at-risk-of-poverty, AROPE);
- the (weak) link between the functional and personal income distributions.

### Extensions and robustness

- self-employment-adjusted labour share (the gross-mixed-income correction);
- cross-country comparison (Portugal vs Spain, Italy, Greece, euro area);
- productivity vs real wages (decoupling) and GVA-based shares;
- redistribution: inequality from market to disposable income.

### Public debt and the interest burden

- general government gross debt (% of GDP) and interest paid (% of GDP and of revenue);
- the implied average interest rate, linking the 2011–2014 debt squeeze to the 2012 break.

## Requirements

```bash
pip install pandas matplotlib dbnomics scipy statsmodels
```

## Run

Open `portugal_gdp_income_distribution.ipynb` in Jupyter, VS Code, or JupyterLab and run all cells. An internet connection is required (data is fetched live from DBnomics).

The notebook exports its tables (the functional-distribution dataset, the statistical-analysis tables, and the living-standards dataset) as CSV files into an `outputs/` directory when executed.
