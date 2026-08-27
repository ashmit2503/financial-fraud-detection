"""Mature-label performance and capacity views."""

import altair as alt
import pandas as pd
import streamlit as st

from app_pages.common import get_data, page_header

data = get_data()
page_header(
    "Performance",
    "Ranking, operating-point, calibration, and review-capacity evidence from mature labels.",
    icon="query_stats",
)

mature = data.batches[
    (data.batches["stream"] == "production") & (data.batches["label_status"] == "mature")
].sort_values("batch_number")
metric = st.segmented_control(
    "Metric",
    options=["pr_auc", "recall", "precision"],
    default="pr_auc",
    format_func=lambda value: value.replace("_", " ").upper(),
)
selected = data.performance[data.performance["metric"] == metric].merge(
    mature[["batch_id", "batch_number"]], on="batch_id", how="inner"
)
line = (
    alt.Chart(selected)
    .mark_line(point=True)
    .encode(
        x=alt.X("batch_number:O", title="Production batch"),
        y=alt.Y("value:Q", title=metric.replace("_", " ").upper(), scale=alt.Scale(zero=False)),
        color=alt.Color(
            "status:N",
            scale=alt.Scale(
                domain=["healthy", "warning", "critical"],
                range=["#34D399", "#FBBF24", "#F87171"],
            ),
            legend=None,
        ),
        tooltip=["batch_id", alt.Tooltip("value:Q", format=".3f"), "status"],
    )
    .properties(height=310)
)
limits = selected.melt(
    id_vars=["batch_number"],
    value_vars=["warning_limit", "critical_limit"],
    var_name="limit",
    value_name="limit_value",
)
rules = (
    alt.Chart(limits)
    .mark_line(strokeDash=[6, 4])
    .encode(
        x=alt.X("batch_number:O"),
        y=alt.Y("limit_value:Q"),
        color=alt.Color(
            "limit:N",
            scale=alt.Scale(
                domain=["warning_limit", "critical_limit"], range=["#FBBF24", "#F87171"]
            ),
            title="Control limit",
        ),
    )
)
st.altair_chart(line + rules, width="stretch")

latest = mature.tail(1).iloc[0]
metrics = st.columns(4)
metrics[0].metric("PR-AUC", float(latest["pr_auc"]), format="%.3f", border=True)
metrics[1].metric("Recall at 2%", float(latest["recall"]), format="percent", border=True)
metrics[2].metric("Precision", float(latest["precision"]), format="percent", border=True)
metrics[3].metric(
    "Captured fraud amount",
    float(latest["captured_fraud_amount_rate"]),
    format="percent",
    border=True,
)

left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.subheader("Latest confusion matrix")
    confusion = pd.DataFrame(
        [
            {
                "Actual": "Legitimate",
                "Predicted legitimate": latest["true_negative"],
                "Predicted fraud": latest["false_positive"],
            },
            {
                "Actual": "Fraud",
                "Predicted legitimate": latest["false_negative"],
                "Predicted fraud": latest["true_positive"],
            },
        ]
    )
    st.dataframe(confusion, hide_index=True, width="stretch")
with right.container(border=True, height="stretch"):
    st.subheader("Calibration over time")
    calibration = mature.melt(
        id_vars=["batch_number"],
        value_vars=["brier_score", "expected_calibration_error"],
        var_name="metric",
        value_name="value",
    )
    calibration_chart = (
        alt.Chart(calibration)
        .mark_line(point=True)
        .encode(
            x=alt.X("batch_number:O", title="Batch"),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("metric:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=["batch_number", "metric", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=220)
    )
    st.altair_chart(calibration_chart, width="stretch")

st.subheader("Acceptance review-budget curve")
if data.review_budgets.empty:
    st.caption("No acceptance review-budget artifact was exported.")
else:
    budget = data.review_budgets.copy()
    budget["target_review_percent"] = budget["target_review_rate"] * 100
    budget_long = budget.melt(
        id_vars=["target_review_percent"],
        value_vars=["precision", "recall", "captured_fraud_amount_rate"],
        var_name="metric",
        value_name="value",
    )
    budget_chart = (
        alt.Chart(budget_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("target_review_percent:Q", title="Review capacity (%)"),
            y=alt.Y("value:Q", title="Rate", axis=alt.Axis(format="%")),
            color=alt.Color("metric:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=[
                alt.Tooltip("target_review_percent:Q", title="Review capacity", format=".1f"),
                "metric",
                alt.Tooltip("value:Q", format=".1%"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(budget_chart, width="stretch")
