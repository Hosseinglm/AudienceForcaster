import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

class DashboardVisuals:
    def __init__(self):
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.template = 'plotly_white'

    def create_time_series_plot(self, df, title="Audience Trends"):
        """Create an advanced time series plot with multiple metrics"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add main audience line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['audience_count'],
                name="Audience Count",
                line=dict(color=self.color_palette[0], width=2),
                fill='tozeroy',
                fillcolor=f'rgba(31, 119, 180, 0.1)'
            ),
            secondary_y=False
        )

        # Add 7-day moving average
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['audience_7day_mean'],
                name="7-Day MA",
                line=dict(color=self.color_palette[1], width=2, dash='dash')
            ),
            secondary_y=False
        )

        # Add engagement score on secondary axis
        if 'engagement_score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['engagement_score'],
                    name="Engagement Score",
                    line=dict(color=self.color_palette[2], width=2),
                    opacity=0.7
                ),
                secondary_y=True
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=24)
            ),
            template=self.template,
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=30, t=100, b=60)
        )

        # Update axes
        fig.update_xaxes(title_text="Date", showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
        fig.update_yaxes(title_text="Audience Count", secondary_y=False)
        fig.update_yaxes(title_text="Engagement Score", secondary_y=True)

        return fig

    def create_engagement_plot(self, df):
        """Create engagement analysis visualization"""
        # Calculate daily averages
        daily_metrics = df.groupby(df['timestamp'].dt.date).agg({
            'EngagementRate': 'mean',
            'ConversionRate': 'mean'
        }).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add engagement rate
        fig.add_trace(
            go.Scatter(
                x=daily_metrics['timestamp'],
                y=daily_metrics['EngagementRate'],
                name="Engagement Rate",
                line=dict(color=self.color_palette[3], width=2)
            ),
            secondary_y=False
        )

        # Add conversion rate
        fig.add_trace(
            go.Scatter(
                x=daily_metrics['timestamp'],
                y=daily_metrics['ConversionRate'],
                name="Conversion Rate",
                line=dict(color=self.color_palette[4], width=2)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title="Engagement and Conversion Trends",
            template=self.template,
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def create_platform_comparison(self, df):
        """Create platform comparison visualization"""
        platform_metrics = df.groupby('platform').agg({
            'audience_count': 'mean',
            'EngagementRate': 'mean',
            'ConversionRate': 'mean'
        }).reset_index()

        fig = px.bar(
            platform_metrics,
            x='platform',
            y=['audience_count', 'EngagementRate', 'ConversionRate'],
            barmode='group',
            title="Platform Performance Comparison",
            template=self.template,
            height=400
        )

        fig.update_layout(
            xaxis_title="Platform",
            yaxis_title="Value",
            legend_title="Metric",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def create_heatmap(self, df):
        """Create audience heatmap by hour and day"""
        # Create hour-day matrix
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()

        heatmap_data = df.pivot_table(
            values='audience_count',
            index='hour',
            columns='day',
            aggfunc='mean'
        )

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis'
        ))

        fig.update_layout(
            title="Audience Heatmap by Hour and Day",
            xaxis_title="Day of Week",
            yaxis_title="Hour of Day",
            height=400,
            template=self.template
        )

        return fig

    def create_forecast_plot(self, historical_df, forecast_series, model_type=""):
        """Create forecast visualization"""
        fig = go.Figure()

        # Historical data
        fig.add_trace(
            go.Scatter(
                x=historical_df.index,
                y=historical_df['audience_count'],
                name='Historical',
                line=dict(color=self.color_palette[0], width=2)
            )
        )

        # Forecast
        forecast_dates = pd.date_range(
            start=historical_df.index[-1] + pd.Timedelta(days=1),
            periods=len(forecast_series)
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_series,
                name=f'{model_type} Forecast',
                line=dict(color=self.color_palette[1], width=2, dash='dash')
            )
        )

        fig.update_layout(
            title=f"Audience Forecast - {model_type}",
            xaxis_title="Date",
            yaxis_title="Audience Count",
            template=self.template,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def create_metrics_dashboard(self, metrics_dict):
        """Create metrics summary visualization"""
        fig = go.Figure()

        for i, (metric, value) in enumerate(metrics_dict.items()):
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=value,
                    title={"text": metric},
                    domain={'row': 0, 'column': i, 'x': [i/len(metrics_dict), (i+1)/len(metrics_dict)]}
                )
            )

        fig.update_layout(
            grid={'rows': 1, 'columns': len(metrics_dict)},
            template=self.template,
            height=200
        )

        return fig