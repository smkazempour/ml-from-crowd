import pandas as pd
import numpy as np

def form_portfolio_from_signals(signals, returns, 
                                stock_col = 'permno', date_col = 'date', 
                                signal_col = 'cap', return_col = 'f_ret',
                                line_up_with = None, shift = 0):

    # Merge signals and returns on stock and date columns
    merged = pd.merge(signals, returns, on=[stock_col, date_col], how='inner', suffixes=('_signal', '_return'))


    if not isinstance(signal_col, list):
        s = signal_col
    else:
        merged['__signal__'] = merged[signal_col].prod(axis=1)
        s = '__signal__'
    
    merged = merged.dropna(subset=[s, return_col]) 
    
    # Create two portfolios based on signals
    long = merged[merged[s] > 0]
    short = merged[merged[s] < 0]
    short[s] = short[s].abs()

    # Calculate weighted returns
    long['__ret x weight__'] = long[return_col] * long[s]
    long = long.groupby(date_col)[['__ret x weight__', s, stock_col]].agg({'__ret x weight__': 'sum', s: 'sum', stock_col: 'count'})
    long['long_ret'] = long['__ret x weight__'] / long[s]
    long.rename(columns={stock_col: 'n_long'}, inplace=True)

    short['__ret x weight__'] = short[return_col] * short[s]
    short = short.groupby(date_col)[['__ret x weight__', s, stock_col]].agg({'__ret x weight__': 'sum', s: 'sum', stock_col: 'count'})
    short['short_ret'] = short['__ret x weight__'] / short[s]
    short.rename(columns={stock_col: 'n_short'}, inplace=True)

    portfolio = pd.merge(long[['long_ret', 'n_long']], short[['short_ret', 'n_short']], left_index=True, right_index=True, how='outer')
    if line_up_with is not None:
        portfolio = portfolio.reindex(line_up_with)
    
    if shift != 0:
        portfolio = portfolio.shift(shift)

    return portfolio


def form_portfolio_from_signals_sort(signals, returns, 
                                     stock_col = 'permno', date_col = 'date', 
                                     signal_col = 'cap', return_col = 'f_ret',
                                     bins = 10, portfolio_weights = None,
                                     line_up_with = None, shift = 0):

    # Merge signals and returns on stock and date columns
    merged = pd.merge(signals, returns, on=[stock_col, date_col], how='inner', suffixes=('_signal', '_return'))


    if not isinstance(signal_col, list):
        s = signal_col
    else:
        merged['__signal__'] = merged[signal_col].prod(axis=1)
        s = '__signal__'
    
    merged = merged.dropna(subset=[s, return_col]) 

    # Create deciles based on signals
    merged['__decile__'] = merged.groupby(date_col)[s].transform(lambda x: pd.qcut(x, bins, labels=False, duplicates='drop') + 1)
    merged['__decile_max__'] = merged.groupby(date_col)['__decile__'].transform('max')
    merged['__decile_min__'] = merged.groupby(date_col)['__decile__'].transform('min')
    if np.any(merged['__decile_max__'] == merged['__decile_min__']):
        print("Warning: Some dates have only one decile. Consider reducing the number of bins or checking for missing values.")

    # Create two portfolios based on signals
    long = merged[merged['__decile__'] == merged['__decile_max__']]
    short = merged[merged['__decile__'] == merged['__decile_min__']]

    # Calculate the weights
    if portfolio_weights is None:
        long['__weight__'] = 1
        short['__weight__'] = 1
    else:
        long['__weight__'] = long[portfolio_weights]
        short['__weight__'] = short[portfolio_weights]
    
    long['__ret x weight__'] = long[return_col] * long['__weight__']
    long = long.groupby(date_col)[['__ret x weight__', '__weight__', stock_col]].agg({'__ret x weight__': 'sum', '__weight__': 'sum', stock_col: 'count'})
    long['long_ret'] = long['__ret x weight__'] / long['__weight__']
    long.rename(columns={stock_col: 'n_long'}, inplace=True)

    short['__ret x weight__'] = short[return_col] * short['__weight__']
    short = short.groupby(date_col)[['__ret x weight__', '__weight__', stock_col]].agg({'__ret x weight__': 'sum', '__weight__': 'sum', stock_col: 'count'})
    short['short_ret'] = short['__ret x weight__'] / short['__weight__']
    short.rename(columns={stock_col: 'n_short'}, inplace=True)

    portfolio = pd.merge(long[['long_ret', 'n_long']], short[['short_ret', 'n_short']], left_index=True, right_index=True, how='outer')

    if line_up_with is not None:
        portfolio = portfolio.reindex(line_up_with)
    
    if shift != 0:
        portfolio = portfolio.shift(shift)

    return portfolio