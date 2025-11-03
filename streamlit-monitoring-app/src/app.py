import streamlit as st
from streamlit_option_menu import option_menu

# Import pages
from pages import detection_status, alerts, system_state

# Set the title of the app
st.title("Live Monitoring Dashboard")

# Create a sidebar for navigation
with st.sidebar:
    selected = option_menu("Main Menu", ["Detection Status", "Alerts", "System State"],
                           icons=["eye", "exclamation-triangle", "dashboard"],
                           menu_icon="cast", default_index=0)

# Render the selected page
if selected == "Detection Status":
    detection_status.show()
elif selected == "Alerts":
    alerts.show()
elif selected == "System State":
    system_state.show()