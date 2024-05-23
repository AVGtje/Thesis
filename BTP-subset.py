import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import matplotlib.dates as mdates

# Path to data
csv_file = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\merged 3.csv"

# Read CSV file
data = pd.read_csv(csv_file, nrows=6144)

# Extract values of y and x1 to x60
y = data['ROR'].values
X = data.drop(['ROR', 'Date'], axis=1).values  # Assuming X is ordered correctly as x1 to x60

print(y)
print(X)

# If you want to use random initialization
x_old = np.random.rand(10)
x_old /= sum(x_old)
x_old = x_old.tolist()
y_old = np.random.rand(6)
y_old /= sum(y_old)
y_old = y_old.tolist()

initial_guess = x_old + y_old

# Adam optimizer from PyTorch
log_params = torch.tensor(list(map(np.log, initial_guess)), dtype=torch.float32, requires_grad=True)
def objective_torch(log_params, X, y):
    params = torch.exp(log_params)  # Exponentiate parameters to ensure they are positive
    a = params[:10]
    b = params[10:16]
    y_pred = sum([a[i] * sum(b[j] * X[:, i*6+j] for j in range(6)) for i in range(10)])
    return torch.sum((y - y_pred) ** 2)
optimizer = torch.optim.Adam([log_params], lr=0.001)

# Minibatch size
batch_size = 32

# Function to create minibatches
def create_minibatches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield X[excerpt], y[excerpt]
import csv
import os

# Define labels for a and b values
a_labels = ['Trend2', 'Trend4', 'Trend8', 'Trend16', 'Trend32', 'Trend64', 'Carry5', 'Carry20', 'Carry60', 'Carry120']
b_labels = ['Agricultural Products', 'Bonds', 'Currencies', 'Energies', 'Equities', 'Metals']

# Optimization loop with minibatches
num_epochs = 20

# Define the directory and file path
directory = r'D:\2022-2023\thesis'
file_path = os.path.join(directory, 'table.csv')

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Prepare the CSV file
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['Time Period'] + a_labels)

    for i in range(10):
        start_time = i * 261
        end_time = 3097 + i * 261
        X_training = X[start_time:end_time]
        y_training = y[start_time:end_time]

        # Convert data to PyTorch tensors
        X_torch = torch.tensor(X_training, dtype=torch.float32)
        y_torch = torch.tensor(y_training, dtype=torch.float32)

        # Initialize parameters in log space to ensure they are positive when exponentiated
        #log_params = torch.tensor(list(map(np.log, initial_guess)), dtype=torch.float32, requires_grad=True)

        for epoch in range(num_epochs):
            for X_batch, y_batch in create_minibatches(X_torch, y_torch, batch_size):
                optimizer.zero_grad()
                loss = objective_torch(log_params, X_batch, y_batch)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                # Apply constraint: sum of b is 1, in exponentiated form
                exp_params = torch.exp(log_params[10:16])
                exp_params /= torch.sum(exp_params)
                log_params[10:16] = torch.log(exp_params)

        print(f"Optimized Parameters for {2000 + i}-{2010 + i}: ", torch.exp(log_params))

        x_new = torch.exp(log_params).tolist()[:10]
        vol_scalar = sum(x_new)
        x_new_normalized = list(map(lambda v: v/vol_scalar, x_new))
        y_new = torch.exp(log_params).tolist()[10:16]

        # Print optimized parameters
        print(f"Optimized vol scalar (normalizing constant) for {2000 + i}-{2010 + i}:")
        print(f"vol scalar: {vol_scalar}")

        print("Optimized a values:")
        for b, a in enumerate(x_new_normalized, start=0):
            print(f"{a_labels[b]}: {a}")

        print("\nOptimized b values:")
        for a, b in enumerate(y_new, start=0):
            print(f"{b_labels[a]}: {b}")

        # Calculate predicted daily return using optimized parameters
        predicted_daily_return = sum([x_new[i] * sum(y_new[j] * X[:, i*6+j] for j in range(6)) for i in range(10)])
        # Calculate R²
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - predicted_daily_return) ** 2)
        r2 = 1 - (ss_res / ss_total)
        print(f"R² for {2000 + i}-{2010 + i}: {r2}")

        # Calculate Sharpe Ratio
        risk_free_rate = 0  # Assuming risk-free rate is 0
        excess_returns = predicted_daily_return - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized Sharpe Ratio
        print(f"Sharpe Ratio for {2000 + i}-{2010 + i}: {sharpe_ratio}")

        # Save the results to CSV
        writer.writerow([f"{2000 + i}-{2010 + i}"] + x_new_normalized)
        
        # Convert daily return to cumulative return
        cumulative_return = np.cumprod(1 + predicted_daily_return)
        ROR = np.cumprod(1 + y)
    
        # Plot cumulative return
        plt.figure()
        plt.plot(cumulative_return, label='Cumulative Return Predicted')
        plt.plot(ROR, label='Cumulative Return SG CTA')
        # Add labels and legend
        plt.xlabel('Day')
        plt.ylabel('Cumulative Return')
        plt.title(f'Cumulative Return with Training Data {2000 + i}-{2010 + i}')
        plt.legend()
        # Show plot
        plt.show()
