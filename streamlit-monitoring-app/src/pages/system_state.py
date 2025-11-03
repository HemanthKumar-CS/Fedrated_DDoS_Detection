from components.metrics import display_metrics
from components.status_indicators import display_status_indicators
from utils.data_processor import load_data
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def show():
    st.title("System State Monitoring")

    # Load data
    data = load_data()

    if data is None:
        st.error("Failed to load system data.")
        return

    # Display system status indicators
    st.header("Current System Status")
    display_status_indicators(data)

    # Display key performance metrics
    st.header("Performance Metrics")
    display_metrics(data)

    # Display alerts if any
    if data.get('alerts'):
        st.header("Alerts")
        for alert in data['alerts']:
            st.warning(alert.get('message', str(alert)))
    else:
        st.success("No alerts at this time.")

    # Display detection status
    st.header("Detection Status")
    if data.get('detection_status'):
        st.write(f"Status: {data['detection_status'].get('status', 'N/A')}")
    else:
        st.success("No detection status available.")
