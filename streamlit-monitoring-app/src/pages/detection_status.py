from src.components.status_indicators import display_detection_status
from src.utils.data_processor import load_detection_data
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
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


if __name__ == "__main__":
    main()
