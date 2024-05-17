# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:13:32 2024

@author: kanr8
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


# Path to data
csv_file = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\merged 3.csv"

# Read CSV file
data = pd.read_csv(csv_file, nrows=6144)

# Extract values of y and x1 to x60
y = data['ROR'].values
X = data.drop('ROR', axis=1).values  # Assuming X is ordered correctly as x1 to x60



# If you want to use equal initialization
# x_old = [equal_starting] * 10
# y_old = [equal_starting/6] * 6

# If you want to use random intialization
x_old = np.random.rand(10)
x_old /= sum(x_old)
x_old = x_old.tolist()
y_old = np.random.rand(6)
y_old /= sum(y_old)
y_old = y_old.tolist()


initial_guess = x_old + y_old


# Choose training subset of data
start_time = 4000
end_time = 6000
X_training = X[start_time:end_time]
y_training = y[start_time:end_time]



# Convert data to PyTorch tensors
X_torch = torch.tensor(X_training, dtype=torch.float32)
y_torch = torch.tensor(y_training, dtype=torch.float32)

# Initialize parameters in log space to ensure they are positive when exponentiated
# (We will perform the optimization in log space to enforce that the parameters are positive.)
log_params = torch.tensor(list(map(np.log, initial_guess)), dtype=torch.float32, requires_grad=True)

# Objective function adapted for PyTorch with exponentiated parameters
def objective_torch(log_params, X, y):
    params = torch.exp(log_params)  # Exponentiate parameters to ensure they are positive
    a = params[:10]
    b = params[10:16]
    y_pred = sum([a[i] * sum(b[j] * X[:, i*6+j] for j in range(6)) for i in range(10)])
    return torch.sum((y - y_pred) ** 2)

# Adam optimizer from PyTorch
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

        
# Optimization loop with minibatches
num_epochs = 1000
for epoch in range(num_epochs):
    for X_batch, y_batch in create_minibatches(X_torch, y_torch, batch_size):
        optimizer.zero_grad()
        loss = objective_torch(log_params, X_batch, y_batch)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        # Apply constraint: sum of b is 1, in exponentiated form
        # This is a bit of a hack, but it works fine, I think
        exp_params = torch.exp(log_params[10:16])
        exp_params /= torch.sum(exp_params)
        log_params[10:16] = torch.log(exp_params)

print("Optimized Parameters:", torch.exp(log_params))


x_new = torch.exp(log_params).tolist()[:10]
vol_scalar = sum(x_new)
x_new_normalized = list(map(lambda v: v/vol_scalar,x_new))
y_new = torch.exp(log_params).tolist()[10:16]

# Print optimized parameters
print("Optimized vol scalar (normalizing constant):")
print(f"vol scalar: {vol_scalar}")

print("Optimized a values:")
for i, a in enumerate(x_new_normalized, start=1):
    print(f"a{i}: {a}")

print("\nOptimized b values:")
for i, b in enumerate(y_new, start=1):
    print(f"b{i}: {b}")

# Calculate predicted daily return using optimized parameters
predicted_daily_return = sum([x_new[i] * sum(y_new[j] * X[:, i*6+j] for j in range(6)) for i in range(10)])
# Convert daily return to cumulative return
cumulative_return = np.cumprod(1 + predicted_daily_return)
ROR= np.cumprod(1 + y)
# Plot cumulative return
plt.plot(cumulative_return, label='Cumulative Return Predicted')
plt.plot(ROR, label='Cumulative Return SG CTA')
# Add labels and legend
plt.xlabel('Day')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Return')
plt.legend()
# Show plot
plt.show()
