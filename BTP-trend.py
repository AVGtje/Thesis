# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
# 1. Data preprocessing: 
#Read adjusted prices and current prices csv 
#After observing we find that the adjusted price and current price csvs have different 
#number of lines of prices. Also within one date, there are more than one prices. So before merging them, 
#we need to preprocess both the adjusted and current prices so that there is a single price for each date 
#and that the 2 csv have the same lines of code.
# Read adjusted prices and current prices CSV files
adjusted_prices_df = pd.read_csv(r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\trend\prices\ALUMINIUM_LME.csv")
current_prices_df = pd.read_csv(r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\carry\multiple_prices-20230911T181856Z-001\multiple_prices\ALUMINIUM_LME.csv")
# Process the date column, keeping only the date part
adjusted_prices_df['DATETIME'] = pd.to_datetime(adjusted_prices_df['DATETIME']).dt.date
current_prices_df['DATETIME'] = pd.to_datetime(current_prices_df['DATETIME']).dt.date
# Merge data for each day and take the average
adjusted_prices_df = adjusted_prices_df.groupby('DATETIME').mean().reset_index()
current_prices_df = current_prices_df.groupby('DATETIME').mean().reset_index()
current_prices_d = current_prices_df[['DATETIME', 'PRICE']]
# Merge both DataFrames using the date as index
prices_df = pd.merge(adjusted_prices_df, current_prices_d, on="DATETIME")
# Check if it works
#print(prices_df)
#2. 
# 2.1 standdeviation
#Loading adjusted and current prices
adjusted_price=prices_df['price']
current_price=prices_df['PRICE']
daily_price_changes = adjusted_price.diff()
percentage_changes = daily_price_changes / current_price.shift(1)
daily_returns = percentage_changes
#return percentage_changes
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
#Check if it works: Yes it does!!!
#print(stdev)
# 2.2 position sizing
# N = (Capital × τ) ÷ (Multiplier × Price × FX × σ %)
    ## resolves to N = (Capital × τ) ÷ (Multiplier × FX × daily stdev price terms × 16)
    ## for simplicity we use the daily risk in price terms, even if we calculated annualised % returns
capital=1000000
risk_target_tau= 0.2
positions_given_variable_risk= capital* risk_target_tau/ ( stdev * (BUSINESS_DAYS_IN_YEAR ** 0.5))
#Check if it works: yes it does!!!!
#print(positions_given_variable_risk)
# 2.3 with trend forecast applied
slow_ewma = adjusted_price.ewm(span=64, min_periods=2).mean()
fast_ewma = adjusted_price.ewm(span=16, min_periods=2).mean()
ewmc=fast_ewma - slow_ewma
#daily_price_vol = stdev_ann_perc.daily_risk_price_terms()
risk_adjusted_ewmac = ewmc / stdev
fast_span=16
scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
forecast_scalar = scalar_dict[fast_span]
scaled_ewmac = risk_adjusted_ewmac * forecast_scalar
capped_ewmac = scaled_ewmac.clip(-20, 20)
position_with_trend_forecast_applied = capped_ewmac * positions_given_variable_risk / 10
#print(position_with_trend_forecast_applied)
#3
# 3.1 Daily percentage return
return_price_points = (adjusted_price - adjusted_price.shift(1))*position_with_trend_forecast_applied.shift(1)
perc_return = return_price_points / capital
#Check if it works: yes it does!!
print(perc_return)
# 3.2 plot
ROR = pd.DataFrame()
ROR['DATE'] = prices_df['DATETIME']
ROR['Daily_return'] = perc_return
#print(ROR)
ROR.set_index('DATE', inplace=True)
plt.figure(figsize=(100, 20))
plt.plot(ROR.index, ROR['Daily_return'], color='blue', marker='o', linestyle='-')
plt.title('Daily Return Over Time')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.xticks(rotation=45)  
plt.grid(True)
plt.tight_layout()
plt.show()
