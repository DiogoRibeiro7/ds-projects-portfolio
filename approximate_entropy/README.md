# Approximate Entropy Notebook Suite

This folder contains five executed Jupyter notebooks that extend the original Approximate Entropy notebook in practical directions.

## Files

1. `01_multiscale_entropy.ipynb`
   - Coarse-graining
   - Multiscale entropy
   - ApEn and SampEn across scales

2. `02_permutation_entropy.ipynb`
   - Ordinal patterns
   - Permutation entropy
   - Comparison with ApEn
   - Rolling entropy through regime changes

3. `03_online_apen_approximation.ipynb`
   - Exact vs approximate rolling ApEn
   - Monte Carlo approximation
   - Streaming-style class interface
   - Runtime and error comparison

4. `04_real_dataset_co2_experiment.ipynb`
   - Real time-series experiment using the `statsmodels` CO2 dataset
   - STL decomposition
   - Rolling ApEn on residual structure

5. `05_regime_detection_pipeline.ipynb`
   - Window feature extraction
   - ApEn and permutation entropy as predictive features
   - Regime classification and entropy-based drift scoring

## Suggested order

Start with 1 and 2 for theory extensions, then 3 for throughput, 4 for a real dataset, and 5 for pipeline integration.

## Notes

- All notebooks are self-contained.
- They were executed before packaging, so plots and tables are already included.
- The real-data notebook uses a built-in offline dataset to keep the suite reproducible.