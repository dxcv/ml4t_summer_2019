"""MC1-P2: Optimize a portfolio. 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
Copyright 2018, Georgia Institute of Technology (Georgia Tech) 			  		 			 	 	 		 		 	  		   	  			  	
Atlanta, Georgia 30332 			  		 			 	 	 		 		 	  		   	  			  	
All Rights Reserved 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
Template code for CS 4646/7646 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
Georgia Tech asserts copyright ownership of this template and all derivative 			  		 			 	 	 		 		 	  		   	  			  	
works, including solutions to the projects assigned in this course. Students 			  		 			 	 	 		 		 	  		   	  			  	
and other users of this template code are advised not to share it with others 			  		 			 	 	 		 		 	  		   	  			  	
or to make it available on publicly viewable websites including repositories 			  		 			 	 	 		 		 	  		   	  			  	
such as github and gitlab.  This copyright statement should not be removed 			  		 			 	 	 		 		 	  		   	  			  	
or edited. 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
We do grant permission to share solutions privately with non-students such 			  		 			 	 	 		 		 	  		   	  			  	
as potential employers. However, sharing with other current or future 			  		 			 	 	 		 		 	  		   	  			  	
students of CS 7646 is prohibited and subject to being investigated as a 			  		 			 	 	 		 		 	  		   	  			  	
GT honor code violation. 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
-----do not edit anything above this line--- 			  		 			 	 	 		 		 	  		   	  			  	
Student Name: Chris Farr
GT User ID: cfarr31
GT ID: 90347082
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
from scipy import optimize as opt
from matplotlib import pyplot as plt


# # TODO Remove this after dev
# import os
# import sys
#
# os.chdir(os.path.join(os.getcwd(), "optimize_something"))
# os.getcwd()


def port_stats(sd, ed,
               syms,
               allocs, rfr=0.0, sf=252.0):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols

    # Handle missing values
    prices = prices.fillna(method="ffill")
    prices = prices.fillna(method="bfill")

    # Get daily portfolio value

    # Normalize (divide stock prices by first row), name norm_price
    norm_price = (prices / prices.iloc[0])
    # Multiply by allocation percentages, allocated_norm_price
    allocated_norm_price = norm_price * allocs
    # Multiply by start_val, name pos_val
    # pos_val = allocated_norm_price * sv
    # Sum each row, name port_val
    port_val = allocated_norm_price.sum(axis=1)

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
    sr = (daily_ret - rfr).mean() / (daily_ret - rfr).std() * np.sqrt(sf)

    return cr, adr, sddr, sr, port_val


def neg_sr(allocs, sd, ed,
           syms, rfr=0.0, sf=252.0):
    cr, adr, sddr, sr, _ = port_stats(sd=sd, ed=ed,
                                      syms=syms,
                                      allocs=allocs, rfr=rfr, sf=sf)
    return -sr


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality 			  		 			 	 	 		 		 	  		   	  			  	
def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):
    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case

    # Initial guess at allocations
    allocs = np.array([1.0 / len(syms) for _ in range(len(syms))])  # TODO add code here to find the allocations

    # Define range and constraints
    range_limit = [(0, 1) for _ in range(len(allocs))]
    constraint = lambda x: np.sum(x) - 1

    # Optimize for
    opt_out = opt.minimize(neg_sr, allocs, args=(sd, ed, syms), bounds=range_limit,
                           constraints={"fun": constraint, "type": "eq"})
    allocs = opt_out.x

    # Compute stats
    cr, adr, sddr, sr, port_val = port_stats(sd, ed, syms, allocs)

    _, _, _, _, spy_val = port_stats(sd, ed, ['SPY'], [1.0])

    # Compare daily portfolio value with SPY using a normalized plot 			  		 			 	 	 		 		 	  		   	  			  	
    if gen_plot:
        # TODO add code to plot here
        df_temp = pd.concat([port_val, spy_val], keys=['Portfolio', 'SPY'], axis=1)
        df_temp.plot()
        plt.title("Daily Portfolio Value and SPY")
        plt.grid(True)
        plt.ylabel("Normalized Price")
        plt.xlabel("Date")
        plt.savefig("optimization_plot.png")

    return allocs, cr, adr, sddr, sr


def test_code():
    # This function WILL NOT be called by the auto grader 			  		 			 	 	 		 		 	  		   	  			  	
    # Do not assume that any variables defined here are available to your function/code 			  		 			 	 	 		 		 	  		   	  			  	
    # It is only here to help you set up and test your code 			  		 			 	 	 		 		 	  		   	  			  	

    # Define input parameters 			  		 			 	 	 		 		 	  		   	  			  	
    # Note that ALL of these values will be set to different values by 			  		 			 	 	 		 		 	  		   	  			  	
    # the autograder!
    end_date = dt.datetime(2010, 12, 31, 0, 0)
    start_date = dt.datetime(2010, 1, 1, 0, 0)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    outputs = allocs = [0.0, 0.4, 0.6, 0.0]

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio 			  		 			 	 	 		 		 	  		   	  			  	
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date,
                                                        syms=symbols,
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
    # This code WILL NOT be called by the auto grader 			  		 			 	 	 		 		 	  		   	  			  	
    # Do not assume that it will be called 			  		 			 	 	 		 		 	  		   	  			  	
    test_code()
