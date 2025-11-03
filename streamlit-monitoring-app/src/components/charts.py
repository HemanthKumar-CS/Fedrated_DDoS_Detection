from typing import List
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def plot_detection_trends(data: pd.DataFrame) -> None:
    st.subheader("Detection Trends")
    fig, ax = plt.subplots()
    ax.plot(data['timestamp'], data['detection_count'], label='Detection Count')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.set_title('Detection Trends Over Time')
    ax.legend()
    st.pyplot(fig)

def plot_alert_frequencies(data: pd.DataFrame) -> None:
    st.subheader("Alert Frequencies")
    alert_counts = data['alert_type'].value_counts()
    fig, ax = plt.subplots()
    alert_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Alert Type')
    ax.set_ylabel('Frequency')
    ax.set_title('Alert Frequencies')
    st.pyplot(fig)

def plot_system_state(data: pd.DataFrame) -> None:
    st.subheader("System State Overview")
    fig, ax = plt.subplots()
    ax.pie(data['state_counts'], labels=data['state_labels'], autopct='%1.1f%%')
    ax.set_title('System State Distribution')
    st.pyplot(fig)