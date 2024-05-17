# -*- coding: utf-8 -*-

# Step 0: Retain only 'Date' and 'Adj Close' for all CSVs
import os
import pandas as pd

# Set directory path
directory = r'D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\OF\Otherfunds_2'

# Iterate through each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        # Read the CSV file
        df = pd.read_csv(filepath)
        # Retain only the 'Date' and 'Adj Close' columns
        df = df[['Date', 'Adj Close']]
        # Save the modified CSV file
        df.to_csv(filepath, index=False)

# Step 0.1: Retain dates until July 19th
import os
import pandas as pd

# Set directory path
directory = r'D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\OF\Otherfunds_0719'

# Iterate through each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        # Read the CSV file
        df = pd.read_csv(filepath)
        # Convert 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        # Retain data before July 20, 2023
        df = df[df['Date'] < '2023-07-20']
        # Modify the 'Date' column format to 'YYYY/MM/DD'
        df['Date'] = df['Date'].dt.strftime('%Y/%m/%d')
        df['ROR'] = df['Adj Close'].pct_change()
        # Save the modified CSV file
        df.to_csv(filepath, index=False)

# Step 1: Calculate daily changes relative to the previous day

# Step 2: Complete each date according to the 'Date' column of r'D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\merged 3.csv'
# Step 2.1: Read the first row date of the current CSV file as first_date and keep merged_df
import os
import pandas as pd

# Set directory path
directory = r'D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\OF\Otherfunds_missing'
merged_filepath = r'D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\merged 3.csv'

# Read the merged 3.csv file
merged_df = pd.read_csv(merged_filepath)
merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%m/%d/%Y')

# Iterate through each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        # Read the CSV file
        df = pd.read_csv(filepath)
        # Convert the 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        # Retain data before July 19, 2023
        df = df[df['Date'] < '2023-07-19']
        # Calculate the daily change rate ROR for each day
        df['ROR'] = df['Adj Close'].pct_change()

        # Get the first row date from the df['Date'] column
        first_date = df['Date'].iloc[0]
        
        # Filter the part of merged_df where the 'Date' column is greater than or equal to first_date
        filtered_merged_df = merged_df[merged_df['Date'] >= first_date].copy()
        
        # Create a dictionary for faster lookup
        ror_dict = df.set_index('Date')['ROR'].to_dict()
        
        # Use lambda expression to generate the new column
        filtered_merged_df['new'] = filtered_merged_df['Date'].apply(
            lambda x: ror_dict[x] if x in ror_dict else None
        )
        
        # Forward fill missing values
        filtered_merged_df['new'].fillna(method='ffill', inplace=True)
        
        # Modify the 'Date' column format to 'YYYY/MM/DD'
        filtered_merged_df['Date'] = filtered_merged_df['Date'].dt.strftime('%Y/%m/%d')

        # Retain only the 'Date' and 'new' columns
        result_df = filtered_merged_df[['Date', 'new']]
        
        # Save the modified CSV file, overwrite the original file
        result_df.to_csv(filepath, index=False)
