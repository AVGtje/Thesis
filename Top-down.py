# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:52:04 2023

@author: kanr8

"""

from math import sqrt
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Load the CSV data with missing values handled
csv_file_path = r"D:\2022-2023\thesis\regression\data\mmmm.csv"#mmmm is a gather of all instrument price data
df = pd.read_csv(csv_file_path, skiprows=[0])  # Skip the first row (column names)
dff = pd.read_csv(csv_file_path)
df.fillna(method='ffill', inplace=True)  # Forward-fill missing values

# Extract the prices of future instruments
prices = df.iloc[:, 1:].apply(pd.to_numeric).values  # Convert to numeric values
prices_vol = df.iloc[:, 1:].apply(pd.to_numeric).rolling(40, min_periods=10).std().values
prices_vol_norm = prices / prices_vol
prices = prices_vol_norm


# Function to calculate correlation sum with missing values handled
def calculate_correlation_sum(selected_indices, prices):
    correlation_sum = 0.0
    for idx in selected_indices:
        corr_values = np.corrcoef(prices[:, idx], prices[:, selected_indices], rowvar=False)
        correlation_sum += np.nansum(corr_values[:-1, -1])  # Ignore NaNs in the correlation calculation
    return correlation_sum

# Function to choose the next instrument with the lowest correlation sum
def choose_next_instrument(selected_indices, prices):
    min_correlation_sum = float('inf')
    next_instrument = None
    for idx in range(prices.shape[1]):
        if idx not in selected_indices:
            temp_indices = selected_indices + [idx]
            correlation_sum = calculate_correlation_sum(temp_indices, prices)
            if correlation_sum < min_correlation_sum:
                min_correlation_sum = correlation_sum
                next_instrument = idx
    return next_instrument

# Main function to select 7 instruments
def select_instruments(prices, num_instruments):
    selected_indices = [np.random.randint(prices.shape[1])]
    while len(selected_indices) < num_instruments:
        next_instrument = choose_next_instrument(selected_indices, prices)
        selected_indices.append(next_instrument)
    return selected_indices

# Call the main function
selected_instruments_indices = select_instruments(prices, num_instruments=27)
selected_instruments_names = dff.columns[selected_instruments_indices] 

s_df=dff[selected_instruments_names]
s_df.fillna(0, inplace=True)
s_df = s_df.astype(float)
# Create a mask of boolean values indicating non-float values
non_float_mask = s_df.applymap(lambda x: not isinstance(x, float))

# Use the mask to count non-float values in each column
non_float_counts = non_float_mask.sum()

#print(non_float_counts)
large_constant = 1e9  

s_df[s_df == np.inf] = large_constant
s_df[s_df == -np.inf] = -large_constant
#print(s_df)
X = s_df.iloc[:, 1:]

# Check for infinity
inf_mask = np.isinf(X)

#print(f"There are {np.sum(inf_mask)} infinite values in X")

#print(X)
print("Selected instruments:")
print(selected_instruments_names)

dfl = pd.read_csv(r"D:\2022-2023\thesis\bottomup\7instr.csv")
dfl = dfl.fillna(0)
date_ROR_df = dfl[['date', 'ROR']]
merged_df = pd.concat([date_ROR_df, s_df], axis=1)
merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df.set_index('date', inplace=True)

window_sizes = [10, 30]
regularizations = ['ElasticNet']
y = merged_df['ROR']
X = merged_df.iloc[:, 1:]

mse_results_monthly = {}
sharpe_results_monthly = {}
r2_results = {}

for window in window_sizes:
    for regularization in regularizations:
        if window > len(merged_df):
            continue
        predicted_ROR = []
        r2_scores = []
        sharpe_ratios = []
        for start in range(len(merged_df) - window):
            X_window = X[start:start + window]
            y_window = y[start:start + window]
            if regularization == 'Lasso':
                model = Lasso(alpha=0.1)
            elif regularization == 'Ridge':
                model = Ridge(alpha=0.1)
            elif regularization == 'ElasticNet':
                model = ElasticNet(alpha=0.000001, l1_ratio=0.5, fit_intercept=False)
            model.fit(X_window, y_window)
            X_next = X.iloc[start + window]
            if not np.isnan(X_next).any():
                prediction = model.predict([X_next])[0]
                predicted_ROR.append(prediction)
        merged_df['predicted_ROR_' + regularization + '_' + str(window)] = pd.Series(predicted_ROR, index=merged_df.index[window:])

        # Resample merged_df to monthly frequency
        merged_df_monthly = merged_df.resample('M').mean()

        # Calculate monthly MSE and store it in the mse_results_monthly dictionary
        mse_monthly = mean_squared_error(merged_df_monthly['ROR'][window:], merged_df_monthly['predicted_ROR_' + regularization + '_' + str(window)][window:])
        mse_results_monthly[regularization + '_' + str(window)] = mse_monthly

        # Calculate and store Sharpe ratio in the sharpe_results_monthly dictionary
        returns = merged_df_monthly['predicted_ROR_' + regularization + '_' + str(window)]
        sharpe_ratio = sqrt(12) * returns.mean() / returns.std()  # assuming that returns are monthly
        sharpe_results_monthly[regularization + '_' + str(window)] = sharpe_ratio

# Print out monthly MSE results and Sharpe ratio results
for key, mse_monthly in mse_results_monthly.items():
    print(f"Monthly MSE for {key}: {mse_monthly}")
    
for key, sharpe_ratio in sharpe_results_monthly.items():
    print(f"Sharpe Ratio for {key}: {sharpe_ratio}")

# Calculate cumulative returns
merged_df_monthly['cumulative_actual_ROR'] = (1 + 10 * 2 * merged_df_monthly['ROR']).cumprod()
for window in window_sizes:
    for regularization in regularizations:
        if window > len(merged_df):
            continue
        column_name = 'predicted_ROR_' + regularization + '_' + str(window)
        merged_df_monthly['cumulative_' + column_name] = (1 + 10 * 2 * merged_df_monthly[column_name]).cumprod()

# Plot cumulative actual and predicted ROR
plt.figure(figsize=(10, 6))
plt.yscale("log")
plt.plot(merged_df_monthly.index, merged_df_monthly['cumulative_actual_ROR'], label='Cumulative Actual ROR (Monthly)')

for window in window_sizes:
    for regularization in regularizations:
        if window > len(merged_df_monthly):
            continue
        column_name = 'cumulative_predicted_ROR_' + regularization + '_' + str(window)
        plt.plot(merged_df_monthly.index, merged_df_monthly[column_name], label='Cumulative Predicted ROR ' + regularization + ', window=' + str(window))

plt.xlabel('Year')
plt.ylabel('Cumulative Return')
plt.legend()
plt.title('Cumulative SG CTA vs the top-down method')
plt.show()
