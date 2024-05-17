
import pandas as pd
import matplotlib.pyplot as plt
def preprocess(adjusted_prices_df, current_prices_df):

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
    current_prices_d = current_prices_df[['DATETIME', 'PRICE','PRICE_CONTRACT','CARRY','CARRY_CONTRACT']]
    # Merge both DataFrames using the date as index
    prices_df = pd.merge(adjusted_prices_df, current_prices_d, on="DATETIME")
    
    return prices_df

#2.1 Standard deviation
def stdev(prices_df):   
    #Loading adjusted and current prices
    adjusted_price=prices_df['price']
    current_price=prices_df['PRICE']
    daily_price_changes = adjusted_price.diff()
    percentage_changes = daily_price_changes / current_price.shift(1)
    daily_returns = percentage_changes
    #return percentage_changes
    daily_exp_std_dev = daily_returns.ewm(span).std()
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
    return stdev

# 2.2 Position sizing
def position_sizing(prices_df, stdev):
    
    # N = (Capital × τ) ÷ (Multiplier × Price × FX × σ %)
    ## resolves to N = (Capital × τ) ÷ (Multiplier × FX × daily stdev price terms × 16)
    ## for simplicity we use the daily risk in price terms, even if we calculated annualised % returns
    capital=1000000
    risk_target_tau= 0.2
    positions_given_variable_risk= capital* risk_target_tau/ ( stdev * (BUSINESS_DAYS_IN_YEAR ** 0.5))
    return positions_given_variable_risk

# 2.3 Position sizing with carry forecast applied
def carry_forecast(prices_df, stdev, positions_given_variable_risk ):
    raw_carry = prices_df['PRICE'] - prices_df['CARRY']
    
    month_from_contract_series = prices_df['CARRY_CONTRACT'].mod(10000)/100.0
    month_as_year_frac_from_contract_series = month_from_contract_series/12.0
    year_from_contract_series = prices_df['CARRY_CONTRACT'].floordiv(10000)
    total_year_frac_from_contract_series = year_from_contract_series + month_as_year_frac_from_contract_series
    
    month_from_contract_series_P = prices_df['PRICE_CONTRACT'].mod(10000)/100.0
    month_as_year_frac_from_contract_series_P = month_from_contract_series_P/12.0
    year_from_contract_series_P = prices_df['CARRY_CONTRACT'].floordiv(10000)
    total_year_frac_from_contract_series_P = year_from_contract_series_P + month_as_year_frac_from_contract_series_P
    
    contract_diff = total_year_frac_from_contract_series - total_year_frac_from_contract_series_P
    ann_carry = raw_carry / contract_diff
    ann_price_vol = stdev
    risk_adj_carry = ann_carry.ffill() / ann_price_vol.ffill()
    smooth_carry = risk_adj_carry.ewm(span).mean()
    scaled_carry = smooth_carry * 30
    capped_carry = scaled_carry.clip(-20, 20)
    return capped_carry

def position_with_carry_forecast_applied_(capped_carry, positions_given_variable_risk):
    position_with_carry_forecast_applied = capped_carry * positions_given_variable_risk / 10
    return position_with_carry_forecast_applied

#3 Results
# 3.1 Daily percentage return
def daily_return_perc(adjusted_price, position_with_carry_forecast_applied):
    
    return_price_points = (adjusted_price - adjusted_price.shift(1))*position_with_carry_forecast_applied.shift(1)
    perc_return = return_price_points / capital
    return perc_return

# 3.2 plot
def plotting(prices_df, perc_return):
    ROR = pd.DataFrame()
    ROR['DATE'] = prices_df['DATETIME']
    ROR['Daily_return'] = perc_return
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

if __name__ == "__main__":

    # 0. Read adjusted prices and current prices CSV files
    adjusted_prices_df = pd.read_csv(r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\trend\prices\ALUMINIUM_LME.csv")
    current_prices_df = pd.read_csv(r"D:\2022-2023\thesis\bottomup\0911\data\individual_instr\indivudual_trendcarry\carry\multiple_prices-20230911T181856Z-001\multiple_prices\ALUMINIUM_LME.csv")
    capital=1000000
    BUSINESS_DAYS_IN_YEAR = 256
    span = 20
    # 1. Data preprocessing: 
    prices_df = preprocess(adjusted_prices_df, current_prices_df)
    adjusted_price = prices_df['price']
    # 2.1 Standard deviation
    stdev = stdev(prices_df)
    # 2.2 Position sizing 
    # N = (Capital × τ) ÷ (Multiplier × Price × FX × σ %)
    positions_given_variable_risk = position_sizing(prices_df, stdev)
    # 2.3 Position sizing with carry forecast applied
    carry_forecast = carry_forecast(prices_df, stdev, positions_given_variable_risk)
    position_with_carry_forecast_applied = position_with_carry_forecast_applied_(carry_forecast, positions_given_variable_risk)
    # 3.1 Daily return
    perc_return = daily_return_perc(adjusted_price, position_with_carry_forecast_applied)
    # 3.2 Plotting
    plotting(prices_df, perc_return)