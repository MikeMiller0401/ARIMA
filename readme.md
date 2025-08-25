# 时间序列建模与预测示例

## 📌 项目简介
本项目演示了如何结合 **线性回归 (Linear Regression)** 与 **ARIMA 模型** 对时间序列数据进行建模与预测。  
主要流程包括：
1. 生成带有趋势和噪声的模拟时间序列数据；
2. 使用线性回归拟合整体趋势；
3. 提取残差并用 ARIMA 建模；
4. 综合趋势预测与残差预测，得到最终未来预测结果；
5. 通过 Matplotlib 绘制不同阶段的可视化图表。

---

## 运行
1. 创建Anaconda虚拟环境：
```
conda create -n ARIMA python=3.8
```

2. 进入环境：
```
conda activate ARIMA
```

3. 进入工程文件夹
```
cd .../.../ARIMA
```

4. 安装依赖
```
pip install -r requirements.txt
```

5. 测试
```
python main.py
```

## 核心流程说明
1. 数据生成
- 使用 np.linspace 构造线性趋势；
- 使用 np.random.normal 生成高斯噪声；
- 构造 pandas.Series，以日期作为索引。

2. 趋势建模

- 使用 LinearRegression 对时间序列进行线性拟合；
- 得到趋势预测值 trend_pred。

3. 残差提取

- 计算残差：residuals = 原始数据 - 拟合趋势。

4. ARIMA 模型

- 在原始序列上拟合 ARIMA(order=(1,1,1))；
- 在残差序列上拟合 ARIMA(order=(1,0,1))。

5. 预测

- 使用线性回归预测未来趋势；
- 使用残差 ARIMA 预测未来噪声；
- 最终预测 = 趋势预测 + 残差预测。

6. 可视化输出
> 生成 3 行 2 列子图：
- 原始数据散点图
- ARIMA 拟合原始序列
- 线性回归 + ARIMA 残差拟合
- 未来预测（带竖线分割训练与预测阶段）
- 残差随时间变化
- 空白占位