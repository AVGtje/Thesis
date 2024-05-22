import os
import pandas as pd

# Input and output folder paths
input_folder_trend = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\trend\prices"
input_folder_carry = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\carry\multiple_prices-20230911T181856Z-001\multiple_prices"
output_folder = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\missingdatahandling\carrywithdate\perc_return_120"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through each CSV file in the prices folder
for filename in os.listdir(input_folder_trend):
    if filename.endswith(".csv"):
        # Read the adjusted prices and current prices CSV files
        adjusted_prices_df = pd.read_csv(os.path.join(input_folder_trend, filename))
        current_prices_df = pd.read_csv(os.path.join(input_folder_carry, filename))
        adjusted_prices_df = adjusted_prices_df.ffill()
        current_prices_df = current_prices_df.ffill()
        span = 120
        # Data preprocessing
        adjusted_prices_df['DATETIME'] = pd.to_datetime(adjusted_prices_df['DATETIME']).dt.date
        current_prices_df['DATETIME'] = pd.to_datetime(current_prices_df['DATETIME']).dt.date

        adjusted_prices_df = adjusted_prices_df.groupby('DATETIME').mean().reset_index()
        current_prices_df = current_prices_df.groupby('DATETIME').mean().reset_index()

        current_prices_d = current_prices_df[['DATETIME', 'PRICE', 'PRICE_CONTRACT', 'CARRY', 'CARRY_CONTRACT']]
        prices_df = pd.merge(adjusted_prices_df, current_prices_d, on="DATETIME")

        # Data processing
        adjusted_price = prices_df['price']
        current_price = prices_df['PRICE']
        daily_price_changes = adjusted_price.diff()
        percentage_changes = daily_price_changes / current_price.shift(1)
        daily_returns = percentage_changes

        daily_exp_std_dev = daily_returns.ewm(span=32).std()
        BUSINESS_DAYS_IN_YEAR = 256
        annualisation_factor = BUSINESS_DAYS_IN_YEAR ** 0.5
        annualised_std_dev = daily_exp_std_dev * annualisation_factor

        ten_year_vol = annualised_std_dev.rolling(BUSINESS_DAYS_IN_YEAR * 10, min_periods=1).mean()
        stdev = 0.3 * ten_year_vol + 0.7 * annualised_std_dev
        stdev = stdev / (BUSINESS_DAYS_IN_YEAR ** 0.5)
        stdev = stdev * current_price

        capital = 1000000
        risk_target_tau = 0.2
        positions_given_variable_risk = capital * risk_target_tau / (stdev * (BUSINESS_DAYS_IN_YEAR ** 0.5))

        raw_carry = prices_df['PRICE'] - prices_df['CARRY']

        month_from_contract_series = prices_df['CARRY_CONTRACT'].mod(10000) / 100.0
        month_as_year_frac_from_contract_series = month_from_contract_series / 12.0
        year_from_contract_series = prices_df['CARRY_CONTRACT'].floordiv(10000)
        total_year_frac_from_contract_series = year_from_contract_series + month_as_year_frac_from_contract_series

        month_from_contract_series_P = prices_df['PRICE_CONTRACT'].mod(10000) / 100.0
        month_as_year_frac_from_contract_series_P = month_from_contract_series_P / 12.0
        year_from_contract_series_P = prices_df['CARRY_CONTRACT'].floordiv(10000)
        total_year_frac_from_contract_series_P = year_from_contract_series_P + month_as_year_frac_from_contract_series_P

        contract_diff = total_year_frac_from_contract_series - total_year_frac_from_contract_series_P
        ann_carry = raw_carry / contract_diff
        ann_price_vol = stdev
        risk_adj_carry = ann_carry.ffill() / ann_price_vol.ffill()
        
        smooth_carry = risk_adj_carry.ewm(span).mean()
        scaled_carry = smooth_carry * 30
        capped_carry = scaled_carry.clip(-20, 20)
        position_with_carry_forecast_applied = capped_carry * positions_given_variable_risk / 10

        return_price_points = (adjusted_price - adjusted_price.shift(1)) * position_with_carry_forecast_applied.shift(1)
        perc_return = return_price_points / capital
        date_df = prices_df[['DATETIME']]  
        perc = pd.concat([date_df, perc_return], axis=1)

        # Save perc_return to a new CSV file
        perc.to_csv(os.path.join(output_folder, f"{filename}"), index=False)
        print(f"Saved perc_return for {filename}")

print("Operation completed!")
