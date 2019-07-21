"""An improved version of your marketsim code that accepts a "trades" data frame (instead of a file). More info on
the trades data frame below. It is OK not to submit this file if you have subsumed its functionality into one of your
other required code files.

Please add in an author function to each file."""


import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def compute_portvals(symbol, trades, start_val=1000000, commission=0, impact=0):

    # Build prices df
    #  Get first and last date from order file
    start_date, end_date = trades.index.min(), trades.index.max()
    #  Get data from get_data function
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices.index = pd.to_datetime(prices.index)
    prices = prices.fillna(method="ffill").fillna(method="bfill")
    #  Add cash column that equals 1
    prices = prices.drop("SPY", axis=1)
    prices["CASH"] = 1
    # Build trades df
    #  Create df: column per symbol, across date range
    # Add cash column to trades
    trades["CASH"] = 0
    trades.loc[start_date, "CASH"] = start_val

    #  Fill in trades by looping through orders file
    for i in trades.index:
        # i = trades.index[5]

        # # Handle trades submitted on non-trading days by going to next trading day
        if i not in prices.index:
            # date = prices.loc[pd.date_range(date, end_date)].dropna().index[0]
            # They said not to fill the order
            continue
        if trades.loc[i, symbol] == 0:
            continue

        trade_vol = trades.loc[i, symbol]
        trade_price = prices.loc[i, symbol]
        impact_adj_trade_price = trade_price * (1+impact) if trade_vol > 0 else trade_price * (1-impact)
        trades.loc[i, "CASH"] -= (trade_vol * impact_adj_trade_price) + commission

    # Build holdings df
    #  Cumulative sum of trades starting at starting_cash (start_val)
    holdings = trades.cumsum()
    # Build values df
    #  Values = prices * holdings
    values = prices * holdings
    # Sum values df to get portfolio value
    port_vals = values.fillna(method="ffill").fillna(method="bfill").sum(axis=1)

    return port_vals.to_frame()


def author():
    return 'cfarr31'  # replace tb34 with your Georgia Tech username.
