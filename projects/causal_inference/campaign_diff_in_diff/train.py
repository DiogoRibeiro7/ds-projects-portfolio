import numpy as np
import pandas as pd
import statsmodels.api as sm


def make_panel_data(
    n_units: int = 200,
    n_periods: int = 8,
    treatment_effect: float = 5.0,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    units = np.arange(n_units)
    periods = np.arange(n_periods)

    data = []
    treated_units = set(rng.choice(units, size=int(n_units * 0.5), replace=False))
    post_periods = set(periods[n_periods // 2 :])

    for unit in units:
        base = rng.normal(50, 5)
        for period in periods:
            is_treated = int(unit in treated_units)
            is_post = int(period in post_periods)
            noise = rng.normal(0, 3)
            outcome = base + 0.8 * period + noise
            if is_treated and is_post:
                outcome += treatment_effect
            data.append((unit, period, is_treated, is_post, outcome))

    return pd.DataFrame(
        data,
        columns=["unit_id", "period", "treated", "post", "outcome"],
    )


def diff_in_diff(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["interaction"] = df["treated"] * df["post"]

    X = df[["treated", "post", "interaction"]]
    X = sm.add_constant(X)
    y = df["outcome"]

    model = sm.OLS(y, X).fit(cov_type="HC1")
    effect = model.params["interaction"]

    return {
        "effect": effect,
        "stderr": model.bse["interaction"],
        "p_value": model.pvalues["interaction"],
        "summary": model.summary().as_text(),
    }


def main() -> None:
    df = make_panel_data()
    results = diff_in_diff(df)

    print("Estimated Treatment Effect (Diff-in-Diff)")
    print(f"Effect: {results['effect']:.2f}")
    print(f"Std Err: {results['stderr']:.2f}")
    print(f"P-value: {results['p_value']:.4f}")


if __name__ == "__main__":
    main()
