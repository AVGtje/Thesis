from math import sqrt
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
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
# From here, you can either continue with prices or prices_vol_norm
# If you are going to regularize, you need to have vol-normalized,
# as all covariates need to be on the same scale.


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
selected_instruments_indices = select_instruments(prices, num_instruments=17)
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


# Load the target ratio of return
dfl = pd.read_csv(r"D:\2022-2023\thesis\bottomup\7instr.csv")
date_ROR_df = dfl['date']
dfl = dfl.ffill()
merged_df = pd.concat([date_ROR_df, s_df], axis=1)
merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df.set_index('date', inplace=True)
other_df = pd.read_csv(r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\OF\Otherfunds_missing\ABYIX.csv", parse_dates=['Date'], dayfirst=False)
other_df = other_df.ffill()
first_date = other_df['Date'].iloc[0]
other_df.set_index('Date', inplace=True)
merged_df = merged_df[merged_df.index >= first_date]
merged_df = pd.concat([other_df['new'],merged_df], axis=1)
print(merged_df)
window_sizes = [20,50,70]
regularizations = ['ElasticNet']
y = other_df['new']
X = merged_df.iloc[:,1:]

r2_results_monthly = {}
sharpe_results_monthly = {}

for window in window_sizes:
    for regularization in regularizations:
        if window > len(merged_df):
            continue
        predicted_ROR = []
        for start in range(len(merged_df) - window):
            X_window = X[start:start + window]
            y_window = y[start:start + window]
            
            if y_window.isna().any() or X_window.isna().any().any():
                y_window = y_window.fillna(y_window.mean())  
                X_window = X_window.fillna(X_window.mean())  
            if regularization == 'Lasso':
                model = Lasso(alpha=0.1, fit_intercept=False)
            elif regularization == 'Ridge':
                model = Ridge(alpha=0.1, fit_intercept=False)
            elif regularization == 'ElasticNet':
                model = ElasticNet(alpha=0.000001, l1_ratio=0.5, fit_intercept=False)
            model.fit(X_window, y_window)
            X_next = X.iloc[start + window]
            if not np.isnan(X_next).any():  
                predicted_ROR.append(model.predict([X_next])[0])
        merged_df['predicted_ROR_' + regularization + '_' + str(window)] = pd.Series(predicted_ROR, index=merged_df.index[window:])
        # Resample merged_df to monthly frequency
        merged_df_monthly = merged_df.resample('M').mean()
        # Calculate  R^2 and store it in the r2_results_monthly dictionary
        r2_monthly = r2_score(merged_df_monthly['new'][window:], merged_df_monthly['predicted_ROR_' + regularization + '_' + str(window)][window:])
        r2_results_monthly[regularization + '_' + str(window)] = r2_monthly

        # Calculate and store the Sharpe ratio in the sharpe_results_monthly dictionary
        returns = merged_df_monthly['predicted_ROR_' + regularization + '_' + str(window)]
        sharpe_ratio = sqrt(12) * returns.mean() / returns.std()  # assuming returns are monthly
        sharpe_results_monthly[regularization + '_' + str(window)] = sharpe_ratio

# Output monthly R^2 results and Sharpe ratio results
for key, r2_monthly in r2_results_monthly.items():
    print(f"Monthly R^2 for {key}: {r2_monthly}")
    
for key, sharpe_ratio in sharpe_results_monthly.items():
    print(f"Sharpe Ratio for {key}: {sharpe_ratio}")


# Calculate cumulative returns
merged_df_monthly['cumulative_actual_ROR'] = (1 + 10*2*merged_df_monthly['new']).cumprod()
for window in window_sizes:
    for regularization in regularizations:
        if window > len(merged_df):
            continue
        column_name = 'predicted_ROR_' + regularization + '_' + str(window)
        merged_df_monthly['cumulative_' + column_name] = (1 + 10*2*merged_df_monthly[column_name]).cumprod()

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
plt.title('Cumulative ABYIX vs the Top-down Method')
plt.show()


