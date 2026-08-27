"""Portfolio fraud monitor overview."""

import altair as alt
import pandas as pd
import streamlit as st

from app_pages.common import get_data, latest_production, page_header, severity_badge

data = get_data()
production = data.batches[data.batches["stream"] == "production"].sort_values("batch_number")
latest = latest_production(data)
stale_count = int((production["label_status"] == "stale").sum())
pending_count = int((production["label_status"] == "pending").sum())
latest_mature = production[production["label_status"] == "mature"].tail(1)

page_header(
    "Fraud model monitor",
    "A frozen fraud model replayed through weekly production batches with delayed labels.",
    icon="shield",
)

with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown("**Current action**")
    current_action = "investigate" if stale_count else str(latest["action"])
    severity_badge(current_action)
    if bool(data.manifest.get("synthetic")):
        st.badge("Synthetic portfolio demo", color="blue")

kpis = st.columns(4)
kpis[0].metric(
    "Review rate",
    float(latest["review_rate"]),
    format="percent",
    border=True,
    icon=":material/rule:",
)
if latest_mature.empty:
    kpis[1].metric("Fraud prevalence", "Unavailable", border=True)
    kpis[2].metric("PR-AUC", "Unavailable", border=True)
else:
    mature = latest_mature.iloc[0]
    kpis[1].metric(
        "Fraud prevalence",
        float(mature["fraud_prevalence"]),
        format="percent",
        border=True,
        icon=":material/report:",
    )
    kpis[2].metric(
        "Latest mature PR-AUC",
        float(mature["pr_auc"]),
        format="%.3f",
        border=True,
        icon=":material/query_stats:",
    )
freshness = "Stale" if stale_count else f"{pending_count} pending" if pending_count else "Current"
kpis[3].metric(
    "Label freshness",
    freshness,
    border=True,
    icon=":material/schedule:",
)

if stale_count:
    st.warning(
        "At least one batch is beyond its expected label-maturity window. Performance evidence "
        "is unavailable until label delivery is repaired.",
        icon=":material/warning:",
    )
elif pending_count:
    st.info(
        f"{pending_count} recent production batches are inside the configured label delay.",
        icon=":material/hourglass_top:",
    )

st.subheader("Health and action timeline")
timeline = production.copy()
timeline["health"] = timeline[["drift_severity", "performance_severity"]].apply(
    lambda row: (
        "critical"
        if "critical" in set(row)
        else "warning"
        if "warning" in set(row)
        else "unavailable"
        if "unavailable" in set(row)
        else "healthy"
    ),
    axis=1,
)
health_order = ["healthy", "unavailable", "warning", "critical"]
chart = (
    alt.Chart(timeline)
    .mark_circle(size=170)
    .encode(
        x=alt.X("batch_number:O", title="Production batch"),
        y=alt.Y("health:N", title=None, sort=health_order),
        color=alt.Color(
            "health:N",
            scale=alt.Scale(
                domain=health_order,
                range=["#34D399", "#94A3B8", "#FBBF24", "#F87171"],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("batch_id:N", title="Batch"),
            alt.Tooltip("label_status:N", title="Labels"),
            alt.Tooltip("action:N", title="Action"),
            alt.Tooltip("review_rate:Q", title="Review rate", format=".2%"),
        ],
    )
    .properties(height=210)
)
st.altair_chart(chart, width="stretch")

left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.subheader("Latest evidence")
    evidence = data.recommendations.sort_values(["stream", "batch_number"]).tail(5).copy()
    evidence["action"] = evidence["action"].str.replace("_", " ")
    st.dataframe(
        evidence[["batch_id", "action", "action_evidence"]],
        hide_index=True,
        width="stretch",
        column_config={
            "batch_id": "Batch",
            "action": "Action",
            "action_evidence": "Evidence",
        },
    )
with right.container(border=True, height="stretch"):
    st.subheader("Deployment contract")
    contract = pd.DataFrame(
        {
            "Field": ["Model version", "Data version", "Review threshold", "Default capacity"],
            "Value": [
                str(latest["model_version"]),
                str(latest["data_version"]),
                f"{float(latest['review_threshold']):.3f}",
                "2%",
            ],
        }
    )
    st.dataframe(contract, hide_index=True, width="stretch")
