from utils.data_processor import load_detection_data
from components.status_indicators import display_detection_status
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def show():
    st.title("Detection Status")

    # Load detection data
    detection_data = load_detection_data()

    if detection_data is not None:
        # Display the current detection status
        display_detection_status(detection_data)

        # Show detailed detection data
        st.subheader("Detailed Detection Data")
        st.dataframe(detection_data)
    else:
        st.error("Failed to load detection data.")
