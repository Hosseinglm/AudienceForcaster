import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_processor import DataProcessor
from src.time_series import TimeSeriesModels
from src.visualizations import DashboardVisuals

# Page configuration
st.set_page_config(
    page_title="Advanced Audience Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0052a3;
        }
        h1, h2, h3 {
            color: #1f2937;
            font-family: 'Segoe UI', sans-serif;
        }
        .stAlert {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 1rem;
        }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Initialize components
data_processor = DataProcessor()
time_series_models = TimeSeriesModels()
visualizer = DashboardVisuals()

# Sidebar
st.sidebar.title("Analytics Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Audience Data (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        # Load and process data
        raw_df = data_processor.load_data(uploaded_file)
        processed_df = data_processor.preprocess_data(raw_df)
        featured_df = data_processor.engineer_features(processed_df)

        # Main content
        st.title("Advanced Audience Analytics Platform")

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Audience", f"{featured_df['audience_count'].sum():,.0f}")
        with col2:
            st.metric("Avg. Engagement Rate", f"{featured_df['EngagementRate'].mean():.2f}%")
        with col3:
            st.metric("Avg. Conversion Rate", f"{featured_df['ConversionRate'].mean():.2f}%")
        with col4:
            st.metric("Forecast Accuracy", f"{100 - featured_df['ForecastError'].mean():.1f}%")

        # Filters
        st.subheader("Data Filters")
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_platform = st.selectbox(
                "Select Platform",
                options=featured_df['platform'].unique()
            )

        with col2:
            min_date = featured_df['timestamp'].min()
            max_date = featured_df['timestamp'].max()
            start_date = st.date_input("Start Date", min_date)

        with col3:
            end_date = st.date_input("End Date", max_date)

        # Filter data based on selection
        mask = (
            (featured_df['timestamp'].dt.date >= start_date) &
            (featured_df['timestamp'].dt.date <= end_date) &
            (featured_df['platform'] == selected_platform)
        )
        filtered_df = featured_df[mask]

        # Prepare data for modeling
        modeling_df = data_processor.prepare_for_modeling(filtered_df)

        # Data Analysis Section
        st.header("Data Analysis")
        tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Platform Insights", "Model Performance"])

        with tab1:
            st.plotly_chart(
                visualizer.create_time_series_plot(
                    modeling_df,
                    "Audience Trends Over Time"
                ),
                use_container_width=True
            )

            # Engagement Analysis
            st.plotly_chart(
                visualizer.create_engagement_plot(filtered_df),
                use_container_width=True
            )

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    visualizer.create_platform_comparison(featured_df),
                    use_container_width=True
                )

            with col2:
                st.plotly_chart(
                    visualizer.create_heatmap(filtered_df),
                    use_container_width=True
                )

        with tab3:
            st.subheader("Model Performance Analysis")

            # Model selection
            model_type = st.selectbox(
                "Select Model",
                ["ARIMA", "Prophet", "SARIMA"]  # Removed LSTM option
            )

            forecast_periods = st.slider(
                "Forecast Horizon (Days)",
                min_value=7,
                max_value=90,
                value=30
            )

            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    # Train model and generate forecast
                    model = time_series_models.train_model(
                        modeling_df,
                        model_type=model_type
                    )

                    forecast = time_series_models.get_forecast(
                        periods=forecast_periods
                    )

                    metrics = time_series_models.calculate_metrics(
                        modeling_df['audience_count'].values,
                        model.fittedvalues
                    )

                    # Display forecast
                    st.plotly_chart(
                        visualizer.create_forecast_plot(
                            modeling_df,
                            forecast,
                            model_type
                        ),
                        use_container_width=True
                    )

                    # Display metrics
                    st.plotly_chart(
                        visualizer.create_metrics_dashboard(metrics),
                        use_container_width=True
                    )

                    # Download forecast
                    forecast_df = pd.DataFrame({
                        'timestamp': pd.date_range(
                            start=modeling_df.index[-1] + timedelta(days=1),
                            periods=forecast_periods
                        ),
                        'forecast': forecast
                    })

                    st.download_button(
                        label="Download Forecast Data",
                        data=forecast_df.to_csv(index=False),
                        file_name=f"{selected_platform}_forecast_{model_type}.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.title("Advanced Audience Analytics Platform")

    # Welcome message
    st.info("""
        👋 Welcome to the Advanced Audience Analytics Platform!

        This platform offers powerful tools for understanding and forecasting audience behavior across various channels.
        With its user-friendly interface and advanced analytics capabilities, you can:
        - Multi-Platform Audience Analysis: 
            Gain insights into your audience's behavior across websites, mobile apps, social media, and more.
        - Advanced Time Series Forecasting: 
            Leverage state-of-the-art models like ARIMA, and Prophet to forecast audience trends.
        - Engagement Metrics Visualization:
            Explore interactive visualizations to understand key metrics such as engagement rates and conversion rates.
        - Model Performance Comparison:
            Compare forecasting models to choose the best-performing one for your needs.
            
        How to Get Started
            Use the sidebar to upload your audience data in CSV format.
            Explore the analytics dashboard to view audience trends and metrics.
            Use the forecasting tools to predict future audience engagement.

    """)