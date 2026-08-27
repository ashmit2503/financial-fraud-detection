"""Investigation evidence, segment errors, and retraining decisions."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from app_pages.common import get_data, page_header, severity_badge

data = get_data()
page_header(
    "Diagnosis and action",
    "Ranked drivers, segment errors, evidence trails, and guarded challenger outcomes.",
    icon="troubleshoot",
)

if data.investigations.empty:
    st.success("No warning or critical investigation records are present.")
    st.stop()

options = list(data.investigations.sort_values("batch_number")["batch_id"])
selected_batch = st.selectbox("Investigation", options, index=len(options) - 1)
record = data.investigations[data.investigations["batch_id"] == selected_batch].iloc[0]

with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown("**Recommended state**")
    severity_badge(str(record["action"]))
    st.markdown("**Label evidence**")
    severity_badge(str(record["label_status"]))

summary = st.columns(3)
summary[0].metric("Likely driver", str(record["likely_driver"]), border=True)
summary[1].metric(
    "Top false-negative segment",
    str(record["top_false_negative_segment"] or "Unavailable"),
    border=True,
)
summary[2].metric(
    "Top prevalence segment",
    str(record["top_prevalence_segment"] or "Unavailable"),
    border=True,
)
st.caption(str(record["classification"]).replace("_", " ").capitalize())
st.warning(str(record["recommended_action"]), icon=":material/assignment:")

left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.subheader("Ranked driver evidence")
    try:
        driver_frame = pd.DataFrame(json.loads(str(record["driver_evidence"])))
    except json.JSONDecodeError:
        driver_frame = pd.DataFrame()
    if driver_frame.empty:
        st.caption("No feature driver could be ranked for this incident.")
    else:
        st.dataframe(driver_frame, hide_index=True, width="stretch")
with right.container(border=True, height="stretch"):
    st.subheader("TreeSHAP importance change")
    shap = (
        data.shap[data.shap["batch_id"] == selected_batch]
        .sort_values("absolute_importance_change", ascending=False)
        .head(10)
    )
    shap_chart = (
        alt.Chart(shap)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", title=None, sort="-x"),
            x=alt.X("importance_change:Q", title="Mean |SHAP| change"),
            color=alt.condition(
                alt.datum.importance_change >= 0,
                alt.value("#F87171"),
                alt.value("#60A5FA"),
            ),
            tooltip=["feature", alt.Tooltip("importance_change:Q", format=".3f")],
        )
        .properties(height=280)
    )
    st.altair_chart(shap_chart, width="stretch")

st.subheader("Segment error contribution")
segments = data.segments[
    (data.segments["batch_id"] == selected_batch) & (data.segments["status"] == "reported")
].sort_values("false_negative", ascending=False)
if segments.empty:
    st.caption("Segment metrics are unavailable or suppressed for insufficient support.")
else:
    st.dataframe(
        segments[
            [
                "segment",
                "segment_value",
                "rows",
                "positives",
                "recall",
                "false_negative",
                "false_positive",
                "fraud_prevalence",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "recall": st.column_config.NumberColumn(format="percent"),
            "fraud_prevalence": st.column_config.NumberColumn(format="percent"),
        },
    )

st.subheader("Recommendation history")
recommendations = data.recommendations[data.recommendations["stream"] == "production"].copy()
recommendations["action"] = recommendations["action"].str.replace("_", " ")
st.dataframe(
    recommendations[
        [
            "batch_id",
            "action",
            "action_evidence",
            "challenger_evaluated",
            "retrain_recommended",
            "challenger_outcome",
        ]
    ].tail(8),
    hide_index=True,
    width="stretch",
)
