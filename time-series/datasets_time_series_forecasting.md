# Datasets for Time Series Forecasting (Energy / Traffic)

This project is designed to work with **public time series datasets** for energy load and traffic volume, ideally with **weather and calendar information** as external regressors.

The notebooks expect a simple, unified format on disk:

- `data/energy.csv` – for energy load experiments  
- `data/traffic.csv` – for traffic volume experiments (optional)

Each file should have at least:

- `timestamp` – datetime in UTC or local time  
- `load` or `volume` – numeric target  
- Optional: weather variables (e.g. `temperature`), holidays, events

Below you will find recommended datasets and how to fetch and preprocess them into this format.

---

## 1. Recommended Energy Datasets

### 1.1 Hourly energy demand, generation and weather (Spain)

**Source:** Kaggle – “Hourly energy demand generation and weather” by Nicholas J. Hana.

**What it contains:**

- ~4 years of **hourly** electrical consumption for Spain.
- Detailed **generation by source**, **market prices**, and **weather** for several cities.
- A central table (often `energy_dataset.csv`) with:
  - A time column (e.g. `time`),
  - Total load columns (e.g. `total load actual`),
  - Weather columns such as `temperature` and `weather description`.

**Why it’s a good default:**

- Already hourly, with clean timestamps.
- Has both **load** and **weather**, ideal for the “external regressors” part of the project.
- Large enough to do proper train/validation/test splits and seasonal analysis.

**How to fetch:**

1. Create or log in to your Kaggle account.
2. Go to the dataset page (search for “Hourly energy demand generation and weather”).
3. Accept the terms if needed and download the files (e.g. `energy_dataset.csv`).

**How to turn it into `data/energy.csv`:**

```python
import pandas as pd
from pathlib import Path

raw = pd.read_csv("energy_dataset.csv")

# Rename columns to the project’s expected schema
df = raw.rename(columns={
    "time": "timestamp",
    "total load actual": "load",
    "temperature": "temperature",
})

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = (
    df.sort_values("timestamp")
      .set_index("timestamp")
      .asfreq("H")              # hourly frequency
      .interpolate(limit_direction="both")  # fill small gaps
)

Path("data").mkdir(exist_ok=True)
df[["load", "temperature"]].to_csv("data/energy.csv")
```

After this, the time series notebook will be able to read `data/energy.csv` directly.

---

### 1.2 Hourly Energy Consumption (PJM, USA)

**Source:** Kaggle – “Hourly Energy Consumption” (PJM regions, US).

**What it contains:**

- Over 10 years of **hourly energy consumption** in megawatts across several PJM regions in the US.
- Each CSV typically has:
  - `Datetime` – timestamp,
  - One column with load (e.g. `PJME_MW` for the PJM East region).

**Why it’s useful:**

- Pure load time series, long history, clear seasonality.
- Good for ARIMA / SARIMA and basic ML without exogenous regressors.

**How to fetch:**

1. Log in to Kaggle.
2. Search for “Hourly Energy Consumption” by Rob Mulla.
3. Download the region file you care about (e.g. `PJME_hourly.csv`).

**How to convert to `data/energy.csv`:**

```python
import pandas as pd
from pathlib import Path

raw = pd.read_csv("PJME_hourly.csv")

df = raw.rename(columns={
    "Datetime": "timestamp",
    "PJME_MW": "load",
})

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = (
    df.sort_values("timestamp")
      .set_index("timestamp")
      .asfreq("H")  # already hourly but this enforces a regular index
)

Path("data").mkdir(exist_ok=True)
df[["load"]].to_csv("data/energy.csv")
```

You lose weather information here, but it’s a clean benchmark for pure univariate forecasting.

---

### 1.3 Weather and Electric Load Dataset

**Source:** Kaggle – “Weather and electric load dataset”.

**What it contains:**

- Hourly load variation plus **hourly weather parameters**.
- Designed explicitly to combine load and weather in a single table.

The preprocessing pattern is similar to the Spain dataset: standardise the timestamp column name, rename the main load and temperature columns, resample to hourly if needed, and save as `data/energy.csv`.

---

## 2. Recommended Traffic Datasets (Optional)

If you want to adapt the notebook to **traffic** forecasting instead of energy, these datasets fit well:

### 2.1 Metro Interstate Traffic Volume

**Source:** Kaggle – “Metro Interstate Traffic Volume”.

**What it contains:**

- **Hourly** traffic volume for a highway segment (I-94 in Minneapolis/St Paul).
- Weather features (temperature, rain/snow, clouds), holiday flags and more.
- Main columns include:
  - `date_time` – timestamp,
  - `traffic_volume` – target,
  - several weather and calendar variables.

**How to use it as `data/traffic.csv`:**

```python
import pandas as pd
from pathlib import Path

raw = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

df = raw.rename(columns={
    "date_time": "timestamp",
    "traffic_volume": "volume",
    # you can also keep temperature or weather cols, e.g. "temp" -> "temperature"
})

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = (
    df.sort_values("timestamp")
      .set_index("timestamp")
      .asfreq("H")
      .interpolate(limit_direction="both")
)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/traffic.csv")
```

You can then either:
- point the notebook to `data/traffic.csv`, or  
- duplicate the energy notebook and swap `load` → `volume` and `energy` → `traffic` in the narrative.

---

## 3. Schema Expected by the Notebooks

For **energy**:

```text
data/energy.csv
- timestamp   (datetime index when loaded)
- load        (float)
- temperature (float, optional)
- ...         (any extra regressors you want to keep)
```

For **traffic** (if you add that):

```text
data/traffic.csv
- timestamp   (datetime index when loaded)
- volume      (float)
- temperature (float, optional)
- weather_*   (categorical or numeric, optional)
```

If your dataset uses different column names, you only need to:

1. Rename them in a small preprocessing script.
2. Ensure the notebook reads the correct file (`energy.csv` or `traffic.csv`).
3. Adjust variable names in the EDA plots if you change `load`/`volume`.

---

## 4. Licensing and Attribution

Each dataset has its own **license** and required **attribution**:

- Kaggle datasets usually list the original data source and licensing terms on the dataset page.  
- UCI datasets (like “Individual Household Electric Power Consumption”) have their own usage notes and citation recommendations.

Before using these data in any **public repo, blog post, or commercial setting**, check:

- The Kaggle / UCI dataset page’s license section.
- Any specified citation format or required notice.

In this project, the notebooks assume you have downloaded and prepared the data **locally**, respecting these terms.

---

## 5. Summary

- Use **Spain hourly energy + weather** or **PJM hourly consumption** as your main energy dataset.
- Optionally, use **Metro Interstate Traffic Volume** for traffic volume forecasting.
- Normalise everything into a simple, regular time index with `timestamp` and a single target (`load` or `volume`), plus any regressors you want.
- Save the result as `data/energy.csv` (and/or `data/traffic.csv`) so the notebooks run without further changes.

