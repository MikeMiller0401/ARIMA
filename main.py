import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# ======================
# 1. Generate 500 time series data points
# ======================
np.random.seed(42)  # 设置随机种子，保证结果可复现
trend = np.linspace(0, 50, 500)  # 模拟线性趋势，从0到50，共500个点
noise = np.random.normal(0, 5, 500)  # 高斯噪声，均值0，标准差5，500个点
data = trend + noise  # 时间序列 = 线性趋势 + 随机噪声
dates = pd.date_range(start="2020-01-01", periods=500, freq="D")  # 生成日期索引，起始2020-01-01，500天
ts = pd.Series(data, index=dates)  # 构造带时间索引的Series对象

# ======================
# 2. Linear regression trend
# ======================
X = np.arange(len(ts)).reshape(-1, 1)  # 样本自变量 = 序列索引(0~499)，reshape(-1,1)转为二维
lr = LinearRegression()  # 初始化线性回归模型
lr.fit(X, ts.values)  # 训练回归模型，输入X（时间索引），输出y（原始序列）
trend_pred = lr.predict(X)  # 拟合出的趋势值


# ======================
# 3. Residuals
# ======================
residuals = ts.values - trend_pred  # 残差 = 原始数据 - 拟合趋势

# ======================
# 4. ARIMA models
# ======================
# ARIMA on original series
# 在原始序列上拟合ARIMA模型
arima_model = ARIMA(ts, order=(1, 1, 1))
# order=(p,d,q)：
#   p=1：AR(自回归)项数
#   d=1：差分阶数，消除非平稳性
#   q=1：MA(滑动平均)项数
arima_results = arima_model.fit()  # 拟合模型

# 在残差上拟合ARIMA模型
res_model = ARIMA(residuals, order=(1, 0, 1))
# d=0 表示残差已经平稳，不需要差分
res_results = res_model.fit()


# ======================
# 5. Forecast next 50 points
# ======================
forecast_steps = 50  # 预测步数（未来50天）
X_future = np.arange(len(ts), len(ts) + forecast_steps).reshape(-1, 1)  # 未来自变量（500~549）
trend_future = lr.predict(X_future)  # 未来趋势预测（线性回归部分）
residuals_future = res_results.forecast(steps=forecast_steps)  # 未来残差预测（ARIMA残差部分）
forecast = trend_future + residuals_future  # 最终预测 = 趋势 + 残差
future_dates = dates[-1] + pd.to_timedelta(np.arange(1, forecast_steps + 1), unit='D')
# 未来日期索引（最后日期+1天，直到+50天）

# ======================
# 6. Create 3x2 subplot
# ======================
fig, axes = plt.subplots(3, 2, figsize=(16, 14)) # 3行2列的子图布局，画布大小16x14

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
