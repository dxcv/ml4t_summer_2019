"""Analyze a portfolio.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

# # TODO Remove this after dev
# import os
# import sys
# os.chdir(os.path.join(os.getcwd(), "optimize_something"))
# os.getcwd()



# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                     syms=['GOOG', 'AAPL', 'GLD', 'XOM'],
                     allocs=[0.1, 0.2, 0.3, 0.4],
                     sv=1000000, rfr=0.0, sf=252.0,
                     gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value

    # Normalize (divide stock prices by first row), name norm_price
    norm_price = (prices / prices.iloc[0])
    # Multiply by allocation percentages, allocated_norm_price
    allocated_norm_price = norm_price * allocs
    # Multiply by start_val, name pos_val
    pos_val = allocated_norm_price * sv
    # Sum each row, name port_val
    port_val = pos_val.sum(axis=1)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    # Calculate...
    # Daily return
    daily_ret = (port_val / port_val.shift(1)) - 1
    daily_ret = daily_ret.iloc[1:]
    # Cumulative return
    cr = port_val.iloc[-1] / port_val.iloc[0] - 1
    # Average daily return
    adr = daily_ret.mean()
    # Std daily return
    sddr = daily_ret.std()
    # Sharpe ratio
    sr = (daily_ret - rfr).mean() / (daily_ret - rfr).std() * np.sqrt(sf)  # TODO this might be wrong

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)

        pass

    # Add code here to properly compute end value
    ev = port_val.iloc[-1]

    return cr, adr, sddr, sr, ev


def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2009, 1, 1)
    end_date = dt.datetime(2010, 1, 1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd=start_date, ed=end_date,
                                             syms=symbols,
                                             allocs=allocations,
                                             sv=start_val,
                                             gen_plot=False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    test_code()
