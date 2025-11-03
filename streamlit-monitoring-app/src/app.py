from utils.data_processor import get_api_health
from pages import detection_status, alerts, system_state
from streamlit_option_menu import option_menu
import streamlit as st
import sys
from pathlib import Path

# Fix imports
sys.path.insert(0, str(Path(__file__).parent))


# Page config
st.set_page_config(page_title="DDoS Detection Dashboard", layout="wide")

# Set the title of the app
st.title("🚨 Federated DDoS Detection Monitoring")

# Check backend connectivity
col1, col2, col3 = st.columns(3)
with col1:
    health = get_api_health()
    if health.get("status") == "healthy":
        st.success("✅ Backend Connected")
    else:
        st.warning("⚠️ Backend Offline")
        st.info("Start with: `python api_service.py`")

# Create a sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    selected = option_menu("Main Menu", ["Detection Status", "Alerts", "System State"],
                           icons=["shield-check", "bell", "activity"],
                           menu_icon="cast", default_index=0)

# Render the selected page
if selected == "Detection Status":
    detection_status.show()
elif selected == "Alerts":
    alerts.show()
elif selected == "System State":
    system_state.show()
