# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:58:01 2024

@author: kanr8
"""

#Step 1 Classifying by asset class.
import os 
import shutil


main_folder = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\trend\trendwithdate_target_cate"


categories = {
    "Equities": ["AEX", "CAC", "DAX", "DOW_mini", "EU-BANKS", "EU-DIV30", "EURO600", "FTSE100", "FTSECHINAA",
                 "MSCIEM-LIFFE", "MSCISING", "NASDAQ_micro", "NIFTY", "NIKKEI", "SP400", "SP500_micro", "TOPIX",
                 "US-DISCRETE", "US-FINANCE", "US-HEALTH", "US-MATERIAL", "US-STAPLES", "US-TECH", "US-UTILS"],
    "Bonds": ["AUSCASH", "BOBL", "BTP", "BTP3", "BUND", "BUXL", "US10", "US2", "US20", "US30", "US5", "GILT", "JGB",
              "SHATZ", "EUROBOR-ICE", "FED", "CADSTIR", "CAD10", "CAD2", "CAD5"],
    "Currencies": ["AUD_micro", "CAD", "CHF", "DX", "EUR_micro", "GBP", "GBPEUR", "JPY", "MXP", "NZD", "YENEUR", "ZAR",
                   "BRE"],
    "Metals": ["AULIMINIUM_LME", "COPPER", "GOLD_micro", "NICKEL_LME", "PALLAD", "PLAT", "SILVER"],
    "Energies": ["BRENT-LAST", "CRUDE_W_mini", "GASOIL", "GASOILINE", "GAS_US_mini", "HEATOIL"],
    "Agricultural Products": ["CANOLA", "COCOA", "COCOA_LDN", "COFFEE", "CORN_mini", "COTTON2", "FEEDCOW", "LEANHOG",
                               "LIVECOW", "MILLWHEAT", "RAPESEED", "REDWHEAT", "ROBUSTA", "RUBBER", "SUGAR11",
                               "SUGAR_WHITE", "WHEAT_mini", "SOYBEAN_mini", "SOYMEAL", "SOYOIL"]
}


# Iterate through each subfolder in the main folder
for subdir in os.listdir(main_folder):
    subdir_path = os.path.join(main_folder, subdir)
    if os.path.isdir(subdir_path):
        # For each subfolder, check if there are CSV files inside
        for root, dirs, files in os.walk(subdir_path):
            for file in files:
                if file.endswith(".csv"):
                    # Get the filename (without extension)
                    filename = os.path.splitext(file)[0]
                    # Iterate through the category list to determine which category the file belongs to
                    for category, items in categories.items():
                        if filename in items:
                            # If a matching category is found, copy the file to the corresponding category folder
                            category_folder = os.path.join(subdir_path, category)
                            os.makedirs(category_folder, exist_ok=True)
                            shutil.copy2(os.path.join(root, file), category_folder)


# Step 2 This is a code to calculate the average value for each asset class under each signal

import os
import csv
import pandas as pd

# Root folder path
base_folder_path = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\error\trendwithdate_target_cate"

# Iterate through all folders under the root folder
for root, dirs, files in os.walk(base_folder_path):
    # Iterate through each folder
    for folder in dirs:
        folder_path = os.path.join(root, folder)
        # Create a new csv file
        new_csv_path = os.path.join(folder_path, "ave.csv")

        # Get all CSV files in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        # Store the second column data for each CSV file
        second_columns = []
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            second_column = df.iloc[:, 1]  # Select the second column
            second_columns.append(second_column)

        # Merge the second column data
        merged_second_column = pd.concat(second_columns, axis=1)
        df = merged_second_column

        # Calculate the average for each row
        df['ave'] = df.mean(axis=1)

        # Save the new CSV file
        df['ave'].to_csv(new_csv_path, index=False)




#Step 3 Merge the average values of all asset class categories under all signals into a single csv

import os
import pandas as pd

# Define the folder path to be merged
folder_path = r'D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\error\opti'

# Create an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Iterate through all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # Read the CSV file, selecting only columns 2 to 7
        data = pd.read_csv(file_path, usecols=[1, 2, 3, 4, 5,6])
        # Add prefix to each column to identify each CSV file
        data.columns = [f'{col}_{file_name[:-4]}' for col in data.columns]
        # Add the read data to the merged DataFrame
        merged_data = pd.concat([merged_data, data], axis=1)

# Save the merged data to a new CSV file
merged_data.to_csv('merged.csv', index=False)
