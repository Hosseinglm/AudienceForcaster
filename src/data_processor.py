import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)

    def load_data(self, uploaded_file):
        """Load and validate uploaded data"""
        try:
            df = pd.read_csv(uploaded_file)

            # Rename columns to match expected format
            column_mapping = {
                'Date': 'timestamp',
                'AudienceCount': 'audience_count',
                'Platform': 'platform'
            }

            # Check if required columns exist
            required_columns = ['Date', 'AudienceCount', 'Platform']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Required: {required_columns}")

            # Rename columns
            df = df.rename(columns=column_mapping)
            return df

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def preprocess_data(self, df):
        """Clean and preprocess the dataset"""
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Handle missing values
        numeric_columns = ['audience_count', 'EngagementRate', 'ConversionRate', 'ForecastError']
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])

        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_prime_time'] = df['hour'].between(19, 22).astype(int)

        return df

    def engineer_features(self, df):
        """Create additional features for modeling"""
        # Rolling statistics
        df['audience_7day_mean'] = df.groupby('platform')['audience_count'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df['audience_7day_std'] = df.groupby('platform')['audience_count'].transform(
            lambda x: x.rolling(7, min_periods=1).std()
        )

        # Add engagement features
        df['engagement_score'] = df['EngagementRate'] * df['ConversionRate']

        # Platform encoding
        df['platform_encoded'] = pd.Categorical(df['platform']).codes

        return df

    def prepare_for_modeling(self, df, target_platform=None):
        """Prepare data for time series modeling"""
        if target_platform:
            df = df[df['platform'] == target_platform].copy()

        # Create features for time series
        modeling_df = df[['timestamp', 'audience_count', 'is_weekend', 'is_prime_time',
                         'audience_7day_mean', 'audience_7day_std', 'engagement_score']].copy()

        modeling_df.set_index('timestamp', inplace=True)
        return modeling_df