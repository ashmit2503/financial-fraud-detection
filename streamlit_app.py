"""Public aggregate-only Streamlit application entry point."""

import streamlit as st

st.set_page_config(
    page_title="Fraud model monitor",
    page_icon=":material/shield:",
    layout="wide",
)

pages = [
    st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("app_pages/performance.py", title="Performance", icon=":material/query_stats:"),
    st.Page("app_pages/drift.py", title="Drift", icon=":material/monitoring:"),
    st.Page(
        "app_pages/diagnosis.py",
        title="Diagnosis and action",
        icon=":material/troubleshoot:",
    ),
]

navigation = st.navigation(pages, position="top")
navigation.run()
