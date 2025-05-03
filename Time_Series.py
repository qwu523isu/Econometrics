# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:20:49 2025

@author: Qiong Wu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch.unitroot import ADF

# Set random seed for reproducibility
np.random.seed(42)

# Generate hypothetical time series: linear trend + random noise
n = 100
time = np.arange(n)
trend = 0.5 * time  # Linear trend
noise = np.random.normal(0, 5, n)  # Random noise
time_series = trend + noise

# Create a pandas Series for convenience
ts_data = pd.Series(time_series, index=pd.date_range('2023-01-01', periods=n, freq='D'))

# Perform ADF test using arch
adf = ADF(ts_data, lags=None, trend='ct', method='AIC')
adf_stat = adf.stat
p_value = adf.pvalue
lags = adf.lags

# Print results
print("Arch ADF Test Results:")
print(f"ADF Statistic: {adf_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Number of lags used: {lags}")

# Interpretation
if p_value < 0.05:
    print("\nConclusion: Reject the null hypothesis (series is stationary).")
else:
    print("\nConclusion: Fail to reject the null hypothesis (series is non-stationary).")

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(ts_data, label='Time Series (Trend + Noise)')
plt.title('Hypothetical Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('time_series_plot.png')
plt.show()