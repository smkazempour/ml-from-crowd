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
                                     stock_col='permno', date_col='date',
                                     signal_col='cap', return_col='f_ret',
                                     bins=10, portfolio_weights=None,
                                     line_up_with=None, shift=0):
    """Fractional ties, cash on constant-signal days, no deletion of modal values."""
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.prediction_metrics import bin_returns

    keys = [stock_col, date_col]
    signal_cols = signal_col if isinstance(signal_col, list) else [signal_col]
    s = signals[keys+signal_cols].copy()
    s["__signal__"] = s[signal_cols].prod(axis=1, min_count=len(signal_cols))
    s = s[keys+["__signal__"]]
    rcols = keys+[return_col]
    if portfolio_weights:
        if portfolio_weights in returns:
            rcols = list(dict.fromkeys(rcols+[portfolio_weights]))
        else:
            s["__weight__"] = signals[portfolio_weights]
    r = returns[rcols].rename(columns={return_col: "__target__"})
    if portfolio_weights and portfolio_weights in returns:
        r = r.rename(columns={portfolio_weights: "__weight__"})
    merged = s.merge(r, on=keys, how="inner", validate="one_to_one")
    merged = merged.rename(columns={date_col: "date"})
    profile, counts = bin_returns(merged, "__signal__", "__target__", bins,
                                 "__weight__" if portfolio_weights else None)
    portfolio = pd.DataFrame({"long_ret": profile[bins], "n_long": counts[bins],
                              "short_ret": profile[1], "n_short": counts[1]})
    portfolio.index.name = date_col
    if line_up_with is not None:
        portfolio = portfolio.reindex(line_up_with)
    return portfolio.shift(shift) if shift else portfolio
