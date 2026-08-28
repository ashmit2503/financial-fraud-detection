"""Shared rendering helpers for Streamlit pages."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from fraud_monitor.dashboard_data import DashboardData, load_dashboard_data

DEFAULT_DEMO_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "demo"


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_load(directory: str) -> DashboardData:
    return load_dashboard_data(directory)


def get_data() -> DashboardData:
    directory = Path(os.environ.get("FRAUD_MONITOR_DEMO_DIR", DEFAULT_DEMO_DIR))
    try:
        return _cached_load(str(directory.resolve()))
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error), icon=":material/error:")
        st.caption("Run `fraud-monitor build-demo --synthetic` to create public demo artifacts.")
        st.stop()


def page_header(title: str, description: str, *, icon: str) -> None:
    st.title(f":material/{icon}: {title}")
    st.caption(description)


def severity_badge(severity: str) -> None:
    colors = {
        "healthy": "green",
        "continue_monitoring": "green",
        "warning": "orange",
        "pending": "orange",
        "stale": "red",
        "critical": "red",
        "investigate": "red",
        "retrain_evaluation_required": "red",
        "unavailable": "gray",
    }
    label = severity.replace("_", " ").capitalize()
    st.badge(label, color=colors.get(severity, "blue"))


def latest_production(data: DashboardData):
    production = data.batches[data.batches["stream"] == "production"]
    if production.empty:
        raise ValueError("Dashboard has no production batches.")
    return production.sort_values("batch_number").iloc[-1]
