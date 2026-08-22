# Customer Segmentation (K-Means)

Segment customers into distinct groups using clustering on synthetic behavioral features.
This project is lightweight and runnable without external datasets.

## What it covers
- Synthetic data generation (behavioral features)
- Feature scaling
- K-Means clustering
- Cluster profiling and summary metrics

## Run
```bash
python projects/machine_learning/customer_segmentation/train.py
```

## Output
- Prints cluster sizes, feature means per cluster, and inertia score.

## Notes
- Replace the synthetic generator with a real customer dataset by loading a CSV
  and updating the feature list in `train.py`.
