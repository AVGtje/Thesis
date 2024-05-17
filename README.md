# Thesis
**Title: A Replication of CTAs Using the Top-Down and Bottom-Up Methods.**

This work consists of two parts, namely, fitting regression to replicate CTAs using the top-down method and the bottom-up method.

**1.The Top-Down Method**

The top-down method uses rolling regression to fit CTA returns. The method is to use linear regressions within a lookback period, using CTA's daily return as training data to predict the next day's ratio of return. Hyperparameters include the rolling time window (which determines the number of training data for each small regression), the number of instruments, and the regularization method (Elasticnet, Lasso, Ridge).

The Top-down.py script implements this method using SG CTA's daily return data from 2000, as explained in detail below.

Data Handling:
The script loads financial instrument price data from a CSV file (mmmm.csv), which is assumed to contain prices with missing values handled using forward-fill method.
The prices are then processed to calculate the ratio of return (ROR) and normalize them by the rolling standard deviation to ensure all covariates are on the same scale.
Feature selection is performed using a custom algorithm to choose a subset of instruments based on their correlation sum.

Model Training and Evaluation:
The script loads the target ROR data from another CSV file (7instr.csv) and combines it with the selected instruments' dataset.
Regularization methods (Lasso, Ridge, ElasticNet) are trained using sliding windows of different sizes and regularization parameters.
Model predictions are made for each window, and the mean squared error (MSE) is calculated as a measure of prediction accuracy.
Cumulative actual and predicted ROR are calculated and plotted for comparison using monthly resampled data.

Results:
The script generates plots showing the cumulative actual ROR and cumulative predicted ROR using different regularization methods and window sizes.

To test the robustness of the top-down method, we run the model with other funds' datasets. The file TPD_otherfunds.py checks if the method can be used to replicate other funds. TPD_otherfunds.py is very similar to Top-down.py except for the input data and y for prediction.


**2. The Bottom-Up Method**

**2.1** 

The bottom-up method consists of the carry strategy and trend strategy. The method uses prices of instruments across 6 asset classes: agricultural products, bonds, energies, metals, currencies, and equities. BTP-carry.py implements the carry strategy using time signals carry5, carry20, carry60, and carry120. The script encompasses various functions for data preprocessing, standard deviation calculation, position sizing, carry forecast, daily return calculation, and plotting.

Functions:

preprocess(adjusted_prices_df, current_prices_df):
Reads adjusted prices and current prices CSV files.
Processes the date column, keeping only the date part.
Merges data for each day and takes the average.

stdev(prices_df):
Calculates the standard deviation of prices.
Uses exponential weighted moving average (EWMA) for volatility estimation.
Computes the annualized standard deviation.

position_sizing(prices_df, stdev):
Computes the position size based on the risk target.
Utilizes the formula: N = (Capital × τ) ÷ (Multiplier × FX × daily stdev price terms × 16).

carry_forecast(prices_df, stdev, positions_given_variable_risk):
Calculates the carry forecast by considering the difference between current and carry prices.
Adjusts for contract expiry and volatility.
Smooths and scales the carry forecast.

position_with_carry_forecast_applied_(capped_carry, positions_given_variable_risk):
Computes the position size with the carry forecast applied.

daily_return_perc(adjusted_price, position_with_carry_forecast_applied):
Computes the daily percentage return.

plotting(prices_df, perc_return):
Plots the daily return over time.

**2.2** 

BTP-trend.py implements the trend strategy using time signals trend2, trend4, trend 8, trend16, trend32, and trend64. Here's an explanation of each part of the code:

Data Preprocessing

Read adjusted prices and current prices from CSV files.
Process the date column, keeping only the date part.
Merge data for each day and take the average.

Standard Deviation

Calculate the daily price changes.
Compute the exponential weighted moving average (EWMA) of daily returns.
Calculate the annualized standard deviation.
Weight with ten-year volatility.
Compute the weighted standard deviation.

Position Sizing

Calculate position size based on given risk target and capital.

Position Sizing with Trend Forecast Applied
Calculate the fast and slow exponential moving averages.
Compute EWMC (Exponential Weighted Moving Center).
Calculate risk-adjusted EWMC.
Scale the risk-adjusted EWMC.
Clip scaled EWMC to set upper and lower limits.
Calculate position size with trend forecast applied.

Results

Compute daily percentage returns.
Plot the daily return over time.

**2.3** 

After getting the daily returns over different instruments and time signals, this part we do an optimization to select the best weights of asset classes and time signals. Bottom-up.py implements this. Description:

The script reads historical data from the specified CSV file.
It initializes parameters for the CTA model by initial_guess.
The parameters are optimized using the Adam optimizer from PyTorch.
The optimized parameters are used to predict daily returns.
Cumulative returns are calculated and plotted against actual cumulative returns.


**2.4.1** 

To test the robustness of the bottom-up method, we select a subset of X and use the weights to predict the remaining subsets. That means we use part of the SG CTA data as training data and the rest of them as testing data. We respectively use 2000-2010, 2001-2011, 2002-2012, 2003-2013, 2004-2014, 2005-2015, 2006-2016, 2007-2017, 2008-2018, 2009-2019, and 2010-2020 as the training dataset. The file BTP-subset.py implements this. It export not only the optimal weights but also cumulative return plots.

**2.4.2** 

To test if we can use bottom-up method to replicate other funds, BTP_otherfunds.py run the optimization with other CTAs' daily return as well as yielding optimzal weights and cumulative plots.

**3. Preprocessing and Data Cleaning**

Before we officially fit the regression, BTP-preprocessing.py preprocesses and cleans input data.

Asset Classification (Step 1):

Firstly, the code iterates through all subfolders in the specified directory.
Then, for each subfolder, it checks if there are any CSV files present.
If CSV files are found, the code matches the filenames (excluding the extension) with predefined asset categories and copies the files to the corresponding category folders.

Calculate Average (Step 2):

The code iterates through all subfolders in the specified directory.
For each subfolder, it reads all CSV files and extracts the second column of data.
Then, it merges the second column data from all CSV files and calculates the average for each row.
Finally, the averages are saved to a new CSV file.

Data Merging (Step 3):

The code reads all CSV files in the specified directory.
For each CSV file, it selects columns 2 to 7 of the data.
Each column name is prefixed to identify which CSV file it came from.
Finally, all the data is merged and saved to a new CSV file named merged.csv.

Before we use other funds datasets to fit regression, Otherfunds_preprocessing preprocess and clean these datasets.

Step 1: Prepare the data by retaining only the 'Date' and 'Adj Close' columns for all CSV files. Additionally, for a specific subset of data, dates are retained only until July 19th, 2023.

Step 2: Calculate daily changes. Daily changes relative to the previous day are calculated for the 'Adj Close' column.

Step 3: Complete missing Dates, match Dates, and complete each date in the financial data according to the 'Date' column of a reference CSV file. 

