import streamlit as st

from src.statistics.core import calculate_power, calculate_sample_size


st.set_page_config(page_title="A/B Test Calculator", layout="centered")

st.title("A/B Test Sample Size & Power Calculator")

st.markdown(
    "Plan experiment size based on baseline rate and minimum detectable effect (MDE)."
)

col1, col2 = st.columns(2)

with col1:
    baseline_rate = st.slider("Baseline conversion rate", 0.01, 0.5, 0.12, 0.01)
    mde = st.slider("Minimum detectable effect (absolute)", 0.001, 0.1, 0.02, 0.001)
    alpha = st.slider("Significance level (alpha)", 0.01, 0.1, 0.05, 0.005)

with col2:
    power = st.slider("Power", 0.6, 0.95, 0.8, 0.01)
    ratio = st.slider("Treatment:Control ratio", 0.5, 3.0, 1.0, 0.1)
    alternative = st.selectbox("Alternative", ["two-sided", "one-sided"])

st.divider()

try:
    n_control = calculate_sample_size(
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative,
    )
    n_treatment = int(n_control * ratio)

    col_a, col_b = st.columns(2)
    col_a.metric("Control sample size", f"{n_control:,}")
    col_b.metric("Treatment sample size", f"{n_treatment:,}")

    achieved_power = calculate_power(
        n_control=n_control,
        n_treatment=n_treatment,
        baseline_rate=baseline_rate,
        effect_size=mde,
        alpha=alpha,
        alternative=alternative,
    )

    st.metric("Estimated power with rounded sizes", f"{achieved_power:.3f}")

except ValueError as exc:
    st.error(f"Input error: {exc}")

st.caption("Uses functions from src.statistics.core for consistency with the main toolkit.")
