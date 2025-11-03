from src.utils.data_processor import load_data
import streamlit as st
from src.components.status_indicators import display_status_indicators
from src.components.metrics import display_metrics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    st.title("System State Monitoring")

    # Load data
    data = load_data()

    # Display system status indicators
    st.header("Current System Status")
    display_status_indicators(data)

    # Display key performance metrics
    st.header("Performance Metrics")
    display_metrics(data)

    # Display alerts if any
    if data['alerts']:
        st.header("Alerts")
        for alert in data['alerts']:
            st.warning(alert)
    else:
        st.success("No alerts at this time.")

    # Display detection status
    st.header("Detection Status")
    if data['detection_status']:
        st.write(data['detection_status'])
    else:
        st.success("No detection status available.")


if __name__ == "__main__":
    main()
