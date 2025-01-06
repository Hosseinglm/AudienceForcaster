import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesModels:
    def __init__(self):
        self.model = None
        self.model_type = None

    def train_model(self, df, model_type="ARIMA"):
        """Train selected time series model"""
        self.model_type = model_type

        if model_type == "ARIMA":
            return self._train_arima(df)
        elif model_type == "SARIMA":
            return self._train_sarima(df)
        elif model_type == "Prophet":
            return self._train_prophet(df)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _train_arima(self, df):
        """Train ARIMA model"""
        self.model = ARIMA(df['audience_count'], order=(2,1,2))
        self.model = self.model.fit()
        return self.model

    def _train_sarima(self, df):
        """Train SARIMA model"""
        self.model = SARIMAX(
            df['audience_count'],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 24)
        )
        self.model = self.model.fit(disp=False)
        return self.model

    def _train_prophet(self, df):
        """Train Prophet model"""
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['audience_count']
        })

        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        self.model.fit(prophet_df)
        return self.model

    def get_forecast(self, periods=30):
        """Generate forecast using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained")

        if self.model_type == "ARIMA" or self.model_type == "SARIMA":
            forecast = self.model.forecast(steps=periods)
            return pd.Series(forecast)

        elif self.model_type == "Prophet":
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            return forecast.tail(periods)['yhat']

    def calculate_metrics(self, actual, predicted):
        """Calculate comprehensive forecast performance metrics"""
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)

        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2
        }