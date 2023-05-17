#%% Useful Libs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
import empyrical as ep
import missingno as msn

plt.style.use("ggplot")

#%% Read Data

# read the Excel file with 3 worksheets
excel_file = pd.ExcelFile('/Users/louishurbin/Downloads/PROJET AAM 2023/DATA.xlsx')

# read the first worksheet into a pandas dataframe
RETURNS = pd.read_excel(excel_file, sheet_name = 'RETURNS') # read excel file

RETURNS['Unnamed: 0'] = pd.to_datetime(RETURNS['Unnamed: 0']) # format excel file
RETURNS.rename(columns = {'Unnamed: 0' : 'date'}, inplace = True)
RETURNS.set_index('date', inplace = True)

RETURNS = RETURNS.apply(lambda x: x.fillna(x.mean()), axis=1) # fill NaNs with cross sectional mean
RETURNS = RETURNS.loc['2007-03':] # start in March 2007


# read the second worksheet into a pandas dataframe
PRICE_TO_BOOK = pd.read_excel(excel_file, sheet_name = 'PRICE TO BOOK') # read excel file

PRICE_TO_BOOK['Unnamed: 0'] = pd.to_datetime(PRICE_TO_BOOK['Unnamed: 0']) # format excel file
PRICE_TO_BOOK.rename(columns = {'Unnamed: 0' : 'date'}, inplace = True)
PRICE_TO_BOOK.set_index('date', inplace = True)

PRICE_TO_BOOK = PRICE_TO_BOOK.apply(lambda x: x.fillna(x.mean()), axis=1) # fill NaNs with cross sectional mean
PRICE_TO_BOOK = PRICE_TO_BOOK.loc['2007-03':] # start in March 2007

# read the third worksheet into a pandas dataframe
BENCHMARK_RETURNS = pd.read_excel(excel_file, sheet_name = 'BENCHMARK RETURNS') # read excel file

BENCHMARK_RETURNS['Unnamed: 0'] = pd.to_datetime(BENCHMARK_RETURNS['Unnamed: 0'])
BENCHMARK_RETURNS.rename(columns = {'Unnamed: 0' : 'date'}, inplace = True)
BENCHMARK_RETURNS.set_index('date', inplace = True)

BENCHMARK_RETURNS = BENCHMARK_RETURNS.loc['2007-03':]  # start in March 2007

#%% Inspect Data

msn.bar(RETURNS)
plt.show()

msn.bar(PRICE_TO_BOOK)
plt.show()

msn.bar(BENCHMARK_RETURNS)
plt.show()

#%% Create DataFrames to store the Stocks's computations

Momentum_RETURNS = RETURNS.shift(1).rolling(window = 11).mean() # average monthly return over the last 12 months but the last one
Value_PRICE_TO_BOOK = 1 / PRICE_TO_BOOK.shift(1) # inverse of the previous Price to Book ratio 


#%% Create DataFrames to store the Z-Score MoM Stock's computations

ZScore_Momentum_RETURNS = zscore(Momentum_RETURNS, axis = 1, ddof=1) # cross-sectional z-score Momentum
ZScore_Value_PRICE_TO_BOOK = zscore(Value_PRICE_TO_BOOK, axis = 1, ddof=1) # cross-sectional z-score P/B

# round to 3 or - 3
ZScore_Momentum_RETURNS = np.clip(ZScore_Momentum_RETURNS, -3, 3) # round to - 3, 3
ZScore_Value_PRICE_TO_BOOK = np.clip(ZScore_Value_PRICE_TO_BOOK, -3, 3) # round to - 3, 3

Global_Score = (ZScore_Momentum_RETURNS + ZScore_Value_PRICE_TO_BOOK)/2 # arithmetical average
Global_Score

#%% Create DataFrames to store both the L/S Portfolio and Individuals Portfolios

Rank_Stock = Global_Score.rank(axis = 1) # rank, top or => 33, bottom are <= 15

Position_Long_Stock = np.where(Rank_Stock >= 33, np.abs(Global_Score), 0) 
Position_Long_Stock = pd.DataFrame(Position_Long_Stock, 
                                    index = Momentum_RETURNS.index, 
                                    columns = Momentum_RETURNS.columns)
Position_Long_Stock = Position_Long_Stock.div(Position_Long_Stock.sum(axis = 1), # weights are > 0 for 15 stocks and the sum is equal to 1
                                              axis = 0)                          # other stocks have a weight of 0 

Position_Short_Stock = np.where(Rank_Stock <= 15, np.abs(Global_Score), 0) 
Position_Short_Stock = pd.DataFrame(Position_Short_Stock, 
                                    index = Momentum_RETURNS.index, 
                                    columns = Momentum_RETURNS.columns)
Position_Short_Stock = Position_Short_Stock.div(Position_Short_Stock.sum(axis = 1), # weights are > 0 for 15 stocks and the sum is equal to 1
                                                axis = 0)                           # other stocks have a weight of 0 

Position_Overall = Position_Long_Stock - Position_Short_Stock # 100% Lon, 100% Short
Position_Overall

#%% Backtest Performance

PnL = pd.DataFrame()
PnL.index = Position_Overall.loc['2008-03':].index

PnL['Monthly'] = np.sum(Position_Overall.loc['2008-03':].shift(1) * RETURNS.loc['2008-03':], axis = 1) # Computation of the PnL using the shift to match [t, t+1[
PnL['Cumulative'] = np.cumsum(PnL['Monthly'])                                                          # Computation of the Cum PnL over the period

alpha_strat = float(LinearRegression().fit(pd.DataFrame(BENCHMARK_RETURNS.loc['2008-03':]), 
                                           pd.DataFrame(PnL['Monthly'])).intercept_)                   # Computation of the Alpha

# Compute alpha
print(f'Alpha: {alpha_strat:.4f}')

# Compute various financial metrics using pyfolio
sortino_ratio = ep.sortino_ratio(PnL['Monthly'], period = 'monthly')
sharpe_ratio = ep.sharpe_ratio(PnL['Monthly'], period = 'monthly', risk_free = 0.02)
max_drawdown = ep.max_drawdown(PnL['Monthly'])

# Print the results
print(f'Sortino ratio: {sortino_ratio:.3f}')
print(f'Sharpe ratio: {sharpe_ratio:.3f}')
print(f'Maximum drawdown: {max_drawdown:.3%}')

#%% Lucky Alpha
N_Sim = 5_000

random_simulations = {'sim_' + str(i+1) : np.clip(pd.DataFrame(data = np.random.normal(0, 1, (len(PnL), 47)), 
                                                             index = PnL.index, 
                                                             columns = Momentum_RETURNS.columns), -3, 3) for i in range( N_Sim)} # Generate a dict of 5K Simulation of weights for each stock for each month, boundaries = [-3,3]
                
random_simulations_rank = {'sim_' + str(i+1) : random_simulations['sim_' + str(i+1)].rank(axis = 1) for i in range(N_Sim)}       # Generate a dict of the ranking for each simulations


random_simulations_long = {'sim_' + str(i+1) : pd.DataFrame(data = np.where(random_simulations_rank['sim_' + str(i+1)] >= 33, 
                                                                          np.abs(random_simulations['sim_' + str(i+1)]), 
                                                                          0), 
                                                                 index = PnL.index, 
                                                                 columns = Momentum_RETURNS.columns) for i in range(N_Sim)}     

random_simulations_long = {'sim_' + str(i+1) : random_simulations_long['sim_' + str(i+1)].div(random_simulations_long['sim_' + str(i+1)].sum(axis = 1),  # Long Portfolio Simulated using a dict
                                                                                          axis = 0) for i in range(N_Sim)}        


random_simulations_short = {'sim_' + str(i+1) : pd.DataFrame(data = np.where(random_simulations_rank['sim_' + str(i+1)] <= 15, 
                                                                          np.abs(random_simulations['sim_' + str(i+1)]), 
                                                                          0), 
                                                                 index = PnL.index, 
                                                                 columns = Momentum_RETURNS.columns) for i in range(N_Sim)}

random_simulations_short = {'sim_' + str(i+1) : random_simulations_short['sim_' + str(i+1)].div(random_simulations_short['sim_' + str(i+1)].sum(axis = 1),  # Short Portfolio Simulated using a dict
                                                                                            axis = 0) for i in range(N_Sim)}

random_global_position = {'sim_' + str(i+1) : random_simulations_long['sim_' + str(i+1)] - random_simulations_short['sim_' + str(i+1)] for i in range(N_Sim)} # L/S Portfolio Construction using both Portfolios

random_backtest_monthly = {'sim_' + str(i+1) : pd.DataFrame(data =  np.sum(random_global_position['sim_' + str(i+1)].shift(1) * RETURNS, axis = 1), 
                                                          index = PnL.index, columns = ['Monthly']) for i in range(N_Sim)}                                    # Dict of the monthly performances for each simulation

random_backtest_alpha = {'sim_' + str(i+1) : float(LinearRegression().fit(BENCHMARK_RETURNS.loc['2008-03':], 
                                                                          random_backtest_monthly['sim_' + str(i+1)]).intercept_) for i in range(N_Sim)}      # Dict to save the Alphas for each simulation

sim_number, sim_alpha = zip(*random_backtest_alpha.items())

estimated_alpha = np.mean(sim_alpha)  # Compute an estimator of the simulated Alphas


#%% Ouput to Excel


# create an Excel writer object
writer = pd.ExcelWriter('Monthly_Portfolio_Weights.xlsx')

# write the DataFrame to an Excel sheet
Position_Overall.dropna(inplace = True)
Position_Overall.to_excel(writer, 
                          index = True,
                          engine='xlsxwriter')

# save the Excel file
writer.save()   # First excel file, 'Monthly_Portfolio_Weights'

# create an Excel writer object
writer = pd.ExcelWriter('Monthly_Performance.xlsx')

# write the DataFrame to an Excel sheet
PnL.dropna(inplace = True)
PnL.to_excel(writer, 
            index = True,
            engine='xlsxwriter')

# save the Excel file
writer.save()   # Second excel file, 'Monthly_Portfolio_Weights'





