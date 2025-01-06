import pandas as pd
import streamlit as st
import base64
import io

def cache_data(func):
    """Decorator for caching data processing results"""
    def wrapper(*args, **kwargs):
        cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = func(*args, **kwargs)
        return st.session_state[cache_key]
    return wrapper

def generate_csv_download_link(df, filename="forecast_results.csv"):
    """Generate a download link for CSV data"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def format_metrics(metrics_dict):
    """Format metrics for display"""
    return pd.DataFrame([metrics_dict]).T.round(2)

def validate_date_range(start_date, end_date):
    """Validate selected date range"""
    if start_date >= end_date:
        st.error("End date must be after start date")
        return False
    return True
