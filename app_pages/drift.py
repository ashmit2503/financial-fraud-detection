"""Feature, prediction, and missingness drift explorer."""

import altair as alt
import streamlit as st

from app_pages.common import get_data, page_header, severity_badge

data = get_data()
page_header(
    "Drift",
    "Empirical feature and score controls fitted only on the locked acceptance reference.",
    icon="monitoring",
)

batch_options = list(
    data.batches.sort_values(["elapsed_day_start", "stream"])["batch_id"].drop_duplicates()
)
selected_batch = st.selectbox("Inspect batch", batch_options, index=len(batch_options) - 1)
batch = data.batches[data.batches["batch_id"] == selected_batch].iloc[0]

with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown("**Overall drift**")
    severity_badge(str(batch["drift_severity"]))
    st.markdown("**Prediction drift**")
    severity_badge(str(batch["prediction_drift_severity"]))

kpis = st.columns(4)
kpis[0].metric("Warning features", int(batch["warning_features"]), border=True)
kpis[1].metric("Critical features", int(batch["critical_features"]), border=True)
kpis[2].metric(
    "Score JS distance", float(batch["score_jensen_shannon"]), format="%.3f", border=True
)
kpis[3].metric(
    "Frozen-threshold review", float(batch["review_rate"]), format="percent", border=True
)

batch_drift = data.feature_drift[data.feature_drift["batch_id"] == selected_batch].copy()
severity_order = {"critical": 0, "warning": 1, "healthy": 2}
batch_drift["severity_order"] = batch_drift["severity"].map(severity_order)
batch_drift = batch_drift.sort_values(["severity_order", "value"], ascending=[True, False])

left, right = st.columns([1.25, 1])
with left.container(border=True, height="stretch"):
    st.subheader("Feature alerts")
    st.dataframe(
        batch_drift[["feature", "metric", "value", "warning_limit", "critical_limit", "severity"]],
        hide_index=True,
        width="stretch",
        column_config={
            "feature": "Feature",
            "metric": "Statistic",
            "value": st.column_config.NumberColumn("Observed", format="%.3f"),
            "warning_limit": st.column_config.NumberColumn("Warning", format="%.3f"),
            "critical_limit": st.column_config.NumberColumn("Critical", format="%.3f"),
            "severity": "Status",
        },
    )
with right.container(border=True, height="stretch"):
    st.subheader("Observed versus limits")
    chart_data = batch_drift.head(12).melt(
        id_vars=["feature"],
        value_vars=["value", "warning_limit", "critical_limit"],
        var_name="series",
        value_name="metric_value",
    )
    comparison = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", title=None, sort="-x"),
            x=alt.X("metric_value:Q", title="Drift statistic"),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom")),
            yOffset="series:N",
            tooltip=["feature", "series", alt.Tooltip("metric_value:Q", format=".3f")],
        )
        .properties(height=300)
    )
    st.altair_chart(comparison, width="stretch")

st.subheader("Prediction score movement")
scores = data.batches.sort_values(["elapsed_day_start", "stream"]).melt(
    id_vars=["batch_id", "elapsed_day_start", "stream"],
    value_vars=["score_mean", "score_p50", "score_p95"],
    var_name="summary",
    value_name="score",
)
score_chart = (
    alt.Chart(scores)
    .mark_line(point=True)
    .encode(
        x=alt.X("elapsed_day_start:Q", title="Elapsed day"),
        y=alt.Y("score:Q", title="Fraud score", scale=alt.Scale(zero=False)),
        color=alt.Color("summary:N", title=None, legend=alt.Legend(orient="bottom")),
        strokeDash=alt.StrokeDash("stream:N", title="Stream"),
        tooltip=["batch_id", "summary", alt.Tooltip("score:Q", format=".3f")],
    )
    .properties(height=290)
)
st.altair_chart(score_chart, width="stretch")
