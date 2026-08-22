# Customer Segmentation Project

This project demonstrates customer segmentation using K-Means clustering and provides an interactive Streamlit dashboard for exploring the results.

## Files
- `pipeline.py`: Runs K-Means clustering on sample customer data.
- `dashboard.py`: Streamlit dashboard to visualize and explore clusters.
- `customers_sample.csv`: Example dataset.

## Usage
1. Run the pipeline to generate clusters:
   ```bash
   python pipeline.py
   ```
2. Launch the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## Requirements
- pandas
- scikit-learn
- streamlit
- plotly

Install requirements with:
```bash
pip install pandas scikit-learn streamlit plotly
```
