import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# ======================
# 1. Generate 500 time series data points
# ======================
np.random.seed(42)
trend = np.linspace(0, 50, 500)
noise = np.random.normal(0, 5, 500)
data = trend + noise
dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
ts = pd.Series(data, index=dates)

# ======================
# 2. Linear regression trend
# ======================
X = np.arange(len(ts)).reshape(-1, 1)
lr = LinearRegression()
lr.fit(X, ts.values)
trend_pred = lr.predict(X)

# ======================
# 3. Residuals
# ======================
residuals = ts.values - trend_pred

# ======================
# 4. ARIMA models
# ======================
# ARIMA on original series
arima_model = ARIMA(ts, order=(1, 1, 1))
arima_results = arima_model.fit()

# ARIMA on residuals
res_model = ARIMA(residuals, order=(1, 0, 1))
res_results = res_model.fit()

# ======================
# 5. Forecast next 50 points
# ======================
forecast_steps = 50
X_future = np.arange(len(ts), len(ts) + forecast_steps).reshape(-1, 1)
trend_future = lr.predict(X_future)
residuals_future = res_results.forecast(steps=forecast_steps)
forecast = trend_future + residuals_future
future_dates = dates[-1] + pd.to_timedelta(np.arange(1, forecast_steps + 1), unit='D')

# ======================
# 6. Create 3x2 subplot
# ======================
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# --- Fig 1: Original data scatter ---
axes[0, 0].scatter(ts.index, ts, color='blue', s=10)
axes[0, 0].set_title('Figure 1: Original Data (Scatter)')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Value')

# --- Fig 2: ARIMA only ---
axes[0, 1].plot(ts.index, arima_results.fittedvalues, color='purple', alpha=0.7)
axes[0, 1].set_title('Figure 2: ARIMA Fitted (Original Series)')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Value')

# --- Fig 3: Linear regression + ARIMA residuals ---
axes[1, 0].plot(ts.index, trend_pred + res_results.fittedvalues, color='red', alpha=0.7)
axes[1, 0].set_title('Figure 3: Linear Regression + ARIMA Residuals')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Value')

# --- Fig 4: Forecast ---
axes[1, 1].plot(ts.index, ts, color='blue', label='Original')
axes[1, 1].plot(future_dates, forecast, color='green', marker='o', label='Forecast')
axes[1, 1].axvline(ts.index[-1], color='black', linestyle='--', label='Forecast Start')
axes[1, 1].set_title('Figure 4: Forecast')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Value')
axes[1, 1].legend()

# --- Fig 5: ARIMA residuals over time ---
axes[2, 0].plot(ts.index, residuals, color='brown', alpha=0.7, label='Residuals')
axes[2, 0].axhline(0, color='black', linestyle='--')
axes[2, 0].set_title('Figure 5: ARIMA Residuals (Original - Trend)')
axes[2, 0].set_xlabel('Date')
axes[2, 0].set_ylabel('Residual Value')
axes[2, 0].legend()

# --- Fig 6: optional / leave empty ---
axes[2, 1].axis('off')  # leave empty

plt.tight_layout()
plt.show()
