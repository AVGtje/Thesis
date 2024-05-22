import os
import pandas as pd

# Input and output folder paths
input_folder_trend = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\trend\prices"
input_folder_carry = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\carry\multiple_prices-20230911T181856Z-001\multiple_prices"
output_folder = r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\trend\trendwithdate\perc_return_2"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through each CSV file in the prices folder
for filename in os.listdir(input_folder_trend):
    if filename.endswith(".csv"):
        adjusted_prices_df = pd.read_csv(os.path.join(input_folder_trend, filename))
        current_prices_df = pd.read_csv(os.path.join(input_folder_carry, filename))
        # Process the date column, keeping only the date part
        adjusted_prices_df['DATETIME'] = pd.to_datetime(adjusted_prices_df['DATETIME']).dt.date
        current_prices_df['DATETIME'] = pd.to_datetime(current_prices_df['DATETIME']).dt.date
        # Merge data for each day and take the average
        adjusted_prices_df = adjusted_prices_df.groupby('DATETIME').mean().reset_index()
        current_prices_df = current_prices_df.groupby('DATETIME').mean().reset_index()
        current_prices_d = current_prices_df[['DATETIME', 'PRICE']]
        # Merge both DataFrames using the date as index
        prices_df = pd.merge(adjusted_prices_df, current_prices_d, on="DATETIME")
       
        # 2.1 standard deviation
        # Loading adjusted and current prices
        adjusted_price = prices_df['price']
        current_price = prices_df['PRICE']
        daily_price_changes = adjusted_price.diff()
        percentage_changes = daily_price_changes / current_price.shift(1)
        daily_returns = percentage_changes
        # return percentage_changes
        daily_exp_std_dev = daily_returns.ewm(span=32).std()
        BUSINESS_DAYS_IN_YEAR = 256
        annualisation_factor = BUSINESS_DAYS_IN_YEAR ** 0.5
        annualised_std_dev = daily_exp_std_dev * annualisation_factor
        ## Weight with ten year vol
        ten_year_vol = annualised_std_dev.rolling(
            BUSINESS_DAYS_IN_YEAR * 10, min_periods=1
        ).mean()
        stdev = 0.3 * ten_year_vol + 0.7 * annualised_std_dev 
        stdev = stdev / (BUSINESS_DAYS_IN_YEAR ** 0.5)
        stdev = stdev * current_price
        
        
        # 2.2 position sizing
        # N = (Capital × τ) ÷ (Multiplier × Price × FX × σ %)
        ## resolves to N = (Capital × τ) ÷ (Multiplier × FX × daily stdev price terms × 16)
        ## for simplicity we use the daily risk in price terms, even if we calculated annualised % returns
        capital = 1000000
        risk_target_tau = 0.2
        positions_given_variable_risk = capital * risk_target_tau / (stdev * (BUSINESS_DAYS_IN_YEAR ** 0.5))

       
        # 2.3 with trend forecast applied
        fast_span = 2
        slow_ewma = adjusted_price.ewm(span=fast_span * 4, min_periods=2).mean()
        fast_ewma = adjusted_price.ewm(span=fast_span, min_periods=2).mean()
        ewmc = fast_ewma - slow_ewma
        # daily_price_vol = stdev_ann_perc.daily_risk_price_terms()
        risk_adjusted_ewmac = ewmc / stdev
        scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
        forecast_scalar = scalar_dict[fast_span]
        scaled_ewmac = risk_adjusted_ewmac * forecast_scalar
        # capped_ewmac = scaled_ewmac.clip(-20, 20)
        position_with_trend_forecast_applied = scaled_ewmac * positions_given_variable_risk / 10
        #print(position_with_trend_forecast_applied)
        # 3.1 Daily percentage return
        return_price_points = (adjusted_price - adjusted_price.shift(1)) * position_with_trend_forecast_applied.shift(1)
        perc_return = return_price_points / capital
        date_df = prices_df[['DATETIME']]
        perc = pd.concat([date_df, perc_return], axis=1)

        # Save perc_return to a new CSV file
        perc.to_csv(os.path.join(output_folder, f"{filename}"), index=False)
        print(f"Saved perc_return for {filename}")

print("Operation completed!")
