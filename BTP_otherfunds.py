import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import matplotlib.dates as mdates

# Path to data
csv_file = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\merged 3.csv"
ROR_file = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\OF\Otherfunds_missing\ABYIX.csv"

# Read CSV file
data = pd.read_csv(csv_file, parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y'))
ROR = pd.read_csv(ROR_file, parse_dates=['Date'], dayfirst=False)


# Fill NaN and infinite values with the values from the previous row
data.fillna(method='ffill', inplace=True)
ROR.fillna(method='ffill', inplace=True)

# Get the first date from the 'Date' column of ROR
first_date = ROR['Date'].iloc[0]
data.set_index('Date', inplace=True)
ROR.set_index('Date', inplace=True)

# Filter out the part in data where the 'Date' column is greater than or equal to the first_date
data = data[data.index >= first_date]

# Extract features and target variables
X = data.drop('ROR', axis=1).values
y = ROR['new'].values

# Check for NaN or infinite values in the data
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
if np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isinf(X)) or np.any(np.isinf(y)):
    raise ValueError("Input data contains NaN or infinite values")

# Random initialization
x_old = np.random.rand(10)
x_old /= sum(x_old)
x_old = x_old.tolist()
y_old = np.random.rand(6)
y_old /= sum(y_old)
y_old = y_old.tolist()

initial_guess = x_old + y_old

# Select training dataset
X_training = X
y_training = y

# Convert data to PyTorch tensors
X_torch = torch.tensor(X_training, dtype=torch.float32)
y_torch = torch.tensor(y_training, dtype=torch.float32)

# Initialize parameters in log space to ensure they are positive when exponentiated
log_params = torch.tensor(list(map(np.log, initial_guess)), dtype=torch.float32, requires_grad=True)

# Objective function adapted for PyTorch with exponentiated parameters and regularization
def objective_torch(log_params, X, y, reg_lambda=1e-4):
    params = torch.exp(log_params)  # Exponentiate parameters to ensure they are positive
    a = params[:10]
    b = params[10:16]
    y_pred = sum([a[i] * sum(b[j] * X[:, i*6+j] for j in range(6)) for i in range(10)])
    loss = torch.sum((y - y_pred) ** 2)
    # Add L2 regularization
    reg_term = reg_lambda * torch.sum(params ** 2)
    return loss + reg_term

# Adam optimizer from PyTorch
optimizer = torch.optim.Adam([log_params], lr=0.0001)

# Minibatch size
batch_size = 32

# Function to create minibatches
def create_minibatches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield X[excerpt], y[excerpt]

# Optimization loop with minibatches and gradient clipping
num_epochs = 1000
for epoch in range(num_epochs):
    for X_batch, y_batch in create_minibatches(X_torch, y_torch, batch_size):
        optimizer.zero_grad()
        loss = objective_torch(log_params, X_batch, y_batch)
        if torch.isnan(loss):
            print("Loss is NaN at epoch:", epoch)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_params], max_norm=1.0)  # Gradient clipping
        optimizer.step()
    with torch.no_grad():
        # Apply constraint: sum of b is 1, in exponentiated form
        exp_params = torch.exp(log_params[10:16])
        exp_params /= torch.sum(exp_params)
        log_params[10:16] = torch.log(exp_params)

    if epoch % 10 == 0:  # Print intermediate results every 10 epochs
        print(f"Epoch {epoch}, Loss: {loss.item()}")


print("Optimized Parameters:", torch.exp(log_params))

x_new = torch.exp(log_params).tolist()[:10]
vol_scalar = sum(x_new)
x_new_normalized = list(map(lambda v: v/vol_scalar, x_new))
y_new = torch.exp(log_params).tolist()[10:16]

# Print optimized parameters
print("Optimized vol scalar (normalizing constant):")
print(f"vol scalar: {vol_scalar}")
'''
print("Optimized a values:")
for i, a in enumerate(x_new_normalized, start=1):
    print(f"a{i}: {a}")

print("\nOptimized b values:")
for i, b in enumerate(y_new, start=1):
    print(f"b{i}: {b}")
'''
print("Optimized a values:")
for i, a in enumerate(x_new_normalized, start=0):
    print(f"{['Carry120', 'Trend16', 'Trend2', 'Carry20', 'Trend32', 'Trend4', 'Carry5', 'Carry60', 'Trend64', 'Trend8'][i-1]}: {a}")

print("\nOptimized b values:")
for i, b in enumerate(y_new, start=0):
    print(f"{['Agricultural Products', 'Bonds', 'Currencies', 'Energies', 'Equities', 'Metals'][i-1]}: {b}")

# Define labels for a values
a_labels = ['Trend2', 'Trend4', 'Trend8', 'Trend16', 'Trend32', 'Trend64', 'Carry5', 'Carry20', 'Carry60', 'Carry120']

# Generate a list of pairs containing labels and corresponding normalized values
a_values_pairs = list(zip(a_labels, x_new_normalized))

# Print the labels and corresponding values
print("Labels and corresponding values for a:")
for label, value in a_values_pairs:
    print(f"{label}: {value}")

# Define labels for b values
b_labels = ['Agricultural Products', 'Bonds', 'Currencies', 'Energies', 'Equities', 'Metals']

# Generate a list of pairs containing labels and corresponding values
b_values_pairs = list(zip(b_labels, y_new))

# Print the labels and corresponding values
print("Labels and corresponding values for b:")
for label, value in b_values_pairs:
    print(f"{label}: {value}")


import csv

# Define the file path and name
csv_file = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\OF\results\ABYIX.csv"

# Write data to CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write header row
    writer.writerow(a_labels + b_labels)
    
    # Write values row
    writer.writerow(x_new_normalized + y_new)
# Calculate predicted daily return using optimized parameters
predicted_daily_return = sum([x_new[i] * sum(y_new[j] * X[:, i*6+j] for j in range(6)) for i in range(10)])
# Calculate R²
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - predicted_daily_return) ** 2)
r2 = 1 - (ss_res / ss_total)
print(f"R²: {r2}")

# Calculate Sharpe Ratio
risk_free_rate = 0  # Assuming risk-free rate is 0
excess_returns = predicted_daily_return - risk_free_rate
sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized Sharpe Ratio
print(f"Sharpe Ratio: {sharpe_ratio}")
# Convert daily return to cumulative return
cumulative_return = np.cumprod(1 + predicted_daily_return)
AB = np.cumprod(1 + y)

plot_data = pd.DataFrame({'Cumulative_Return_Predicted': cumulative_return, 'Cumulative_Return_ABYIX': AB}, index=ROR.index)


fig, ax = plt.subplots()
plt.plot(plot_data.index, plot_data['Cumulative_Return_Predicted'], label='Cumulative Return Predicted')
plt.plot(plot_data.index, plot_data['Cumulative_Return_ABYIX'], label='Cumulative Return ABYIX')


plt.xlabel('Year')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Return ABYIX and the Bottom-up Method')
plt.legend()
plt.xticks(rotation=45)
