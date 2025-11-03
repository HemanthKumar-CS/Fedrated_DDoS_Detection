from utils.data_processor import load_alerts
import streamlit as st
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def display_alerts(alerts):
    for alert in alerts:
        st.markdown(f"### Alert: {alert['title']}")
        st.markdown(f"**Description:** {alert['description']}")
        st.markdown(f"**Severity:** {alert['severity']}")
        st.markdown(f"**Timestamp:** {alert['timestamp']}")
        st.markdown("---")


def show():
    st.title("Alerts Dashboard")

    # Load alerts from JSON data
    alerts = load_alerts()

    if alerts:
        display_alerts(alerts)
    else:
        st.write("No alerts at this time.")
