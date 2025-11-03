from datetime import datetime
import streamlit as st

def display_metrics(detection_count, alert_count, normal_conditions):
    st.subheader("Key Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Detection Count", value=detection_count, delta=detection_count - (detection_count - 1))
    
    with col2:
        st.metric(label="Alert Count", value=alert_count, delta=alert_count - (alert_count - 1))
    
    with col3:
        st.metric(label="Normal Conditions", value=normal_conditions, delta=normal_conditions - (normal_conditions - 1))

def update_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")