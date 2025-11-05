# Telco Customer Churn – End-to-End Project

This repository contains an end-to-end **customer churn** project built around the
IBM **Telco Customer Churn** dataset. The project is organised as a trilogy of
Jupyter notebooks:

1. **Notebook 1 – Core churn project**
   Business framing, EDA, baseline and main models, feature importance.

2. **Notebook 2 – Model tuning & cost-sensitive decisions**
   Hyperparameter tuning with cross-validation and business-aware threshold selection.

3. **Notebook 3 – Monitoring & drift analysis (simulated)**
   Simulated production monitoring, score drift (PSI), and simple alert logic.

The goal is not only to build a good model, but also to show how you would **operate**
and **reason about** a churn model in a realistic setting.

---

## 1. Project structure

```text
.
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # dataset (not committed)
├── telco_churn_project.ipynb                 # Notebook 1 – core project
├── telco_churn_tuning_and_costs.ipynb        # Notebook 2 – tuning & cost
├── telco_churn_monitoring_and_drift.ipynb    # Notebook 3 – monitoring & drift
└── README.md
```

* The `data/` folder is **not** versioned. You need to download the CSV yourself.
* All notebooks assume the dataset is available under the exact path above.

---

## 2. Dataset

We use the public **IBM Telco Customer Churn** dataset (commonly found on Kaggle
and IBM sample galleries).

**Target variable**

* `Churn` – `"Yes"` if the customer left, `"No"` otherwise.

**Example features**

* Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
* Contract & billing: `Contract`, `PaperlessBilling`, `PaymentMethod`.
* Services: `PhoneService`, `InternetService`, `StreamingTV`, etc.
* Continuous: `tenure`, `MonthlyCharges`, `TotalCharges`.

### How to obtain the data

1. Download the CSV (e.g. from Kaggle).

2. Create a local folder:

   ```bash
   mkdir -p data
   ```

3. Save the file as:

   ```text
   data/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

No other changes are needed – the notebooks expect this location.

---

## 3. Environment & requirements

You can use any recent Python 3 environment (3.9+ recommended).

Minimal package list:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `jupyter` or `jupyterlab`

Example installation with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Then start Jupyter:

```bash
jupyter lab
# or
jupyter notebook
```

---

## 4. Notebooks overview

### 4.1 Notebook 1 – Core churn project

**File:** `telco_churn_project.ipynb`

Focus:

* Frame churn as a **business problem** (why churn matters, what we want to do).
* Load and clean the Telco dataset:

  * Convert `TotalCharges` to numeric and handle missing values.
  * Remove duplicate `customerID`s.
* Exploratory Data Analysis (EDA):

  * Churn rate and class balance.
  * Distributions of `tenure`, `MonthlyCharges`, `TotalCharges` by churn status.
  * Churn rates by contract, internet service, payment method, etc.
* Modelling:

  * Train–test split with stratification.
  * Preprocessing pipeline using `ColumnTransformer`:

    * Standard scaling for numeric features.
    * One-hot encoding for categorical features.
  * Models:

    * `DummyClassifier` baseline (most frequent class).
    * Logistic Regression.
    * Random Forest.
* Evaluation:

  * Accuracy, ROC-AUC, classification report.
  * Confusion matrix and ROC curves.
  * Comparative table of model metrics.
* Interpretation:

  * Random Forest feature importance.
  * Discussion of key churn drivers and possible business actions.

The notebook is written in a **Kaggle-style** format with markdown explanations
before and after each major code block.

---

### 4.2 Notebook 2 – Tuning & cost-sensitive decisions

**File:** `telco_churn_tuning_and_costs.ipynb`

Focus:

* Re-load and clean the dataset (notebook is self-contained).
* Build the same preprocessing pipeline (scaling + one-hot encoding).
* Define a general evaluation helper for classifiers.
* Train a **dummy baseline** for reference.

**Hyperparameter tuning**

* Use `RandomizedSearchCV` with cross-validation and `roc_auc` scoring to tune:

  * Logistic Regression (regularisation strength `C`, etc.).
  * Random Forest (number of trees, depth, min samples split/leaf, max features).
* Inspect:

  * Best parameters.
  * Cross-validated ROC-AUC.
  * Test performance of the tuned models.

**Cost-sensitive threshold selection**

* Introduce a simple cost model:

  * `C_FP`: cost of contacting a non-churner.
  * `C_FN`: cost of failing to identify a churner.
* For the tuned Random Forest:

  * Evaluate multiple probability thresholds.
  * Compute confusion matrix and **expected cost per customer** for each threshold.
  * Identify the threshold that minimises cost.
* Visualisations:

  * Cost per customer vs threshold.
  * Precision–recall curve.

Outcome: you end up with a **tuned model + business-aware decision threshold**, not
just a default `0.5` cut-off.

---

### 4.3 Notebook 3 – Monitoring & drift analysis

**File:** `telco_churn_monitoring_and_drift.ipynb`

Focus:

* Rebuild the Telco churn model (Random Forest with preprocessing).
* Treat it as a **production model** and focus on monitoring rather than tuning.

**Simulated monitoring windows**

* Shuffle the test set and split it into a number of **time windows**.
* For each window, compute:

  * Accuracy.
  * ROC-AUC.
  * Observed churn rate.
  * Predicted positive rate.
* Plot metrics over windows to mimic **time series monitoring**.

**Drift detection with PSI**

* Use training score distribution as the **reference**.
* For each window, compute the **Population Stability Index (PSI)** on model scores.
* Plot PSI vs window with rule-of-thumb thresholds (0.1, 0.25).
* Interpret possible drift and what it would mean for a live system.

**Simple alert logic**

* Raise a warning if:

  * ROC-AUC in a window falls below a chosen threshold.
  * PSI in a window exceeds a drift threshold.
* Discuss how this would plug into real alerting and retraining pipelines.

This notebook gives a compact example of how to **operate and monitor** a churn
model over time.

---

## 5. Suggested order of use

1. **Start with Notebook 1**
   Get familiar with the data, modelling pipeline, and key churn drivers.

2. **Move to Notebook 2**
   Improve model performance and decide on a meaningful probability threshold
   given business costs.

3. **Finish with Notebook 3**
   Think about how the churn model behaves **after deployment** – monitoring,
   drift, and alerts.

---

## 6. Possible extensions

Ideas for extending this project:

* Add **other algorithms** (Gradient Boosting, XGBoost, LightGBM, CatBoost).
* Use **calibrated probabilities** and more detailed cost models (different
  costs by segment or customer value).
* Integrate with a simple **API** (e.g. FastAPI) for scoring.
* Store predictions and monitoring metrics in a database or data lake.
* Build a small **dashboard** for churn risk and monitoring.
* Replace Telco data with **SaaS or subscription churn** and adapt the features.

---

## 7. License and usage

* The notebooks are intended for **educational and demonstration** purposes.
* The Telco dataset is owned and licensed by IBM / the respective source (e.g. Kaggle);
  please refer to the original dataset page for licence terms before using it
  in production or commercial contexts.
