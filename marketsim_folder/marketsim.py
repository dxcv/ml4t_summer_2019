"""MC2-P1: Market simulator. 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
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
import os
from util import get_data, plot_data

# os.chdir(os.path.join(os.getcwd(), "marketsim_folder"))

"""
Commissions: For each trade that you execute, charge a commission according to the parameter sent. Treat that as a 
deduction from your cash balance. 

Market impact: For each trade that you execute, assume that the stock price moves 
against you according to the impact parameter. So if you are buying, assume the price goes up before your purchase. 
Similarly, if selling, assume the price drops 50 bps before the sale. For simplicity treat the market impact penalty 
as a deduction from your cash balance. 
"""


def compute_portvals(orders_file="./orders/orders-01.csv", start_val=1000000, commission=9.95, impact=0.005):

    # orders_file = "./additional_orders/orders-short.csv"
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your 			  		 			 	 	 		 		 	  		   	  			  	
    # code should work correctly with either input 			  		 			 	 	 		 		 	  		   	  			  	
    # TODO: Your code here
    # os.getcwd()
    # Read orders_file if string
    orders = pd.read_csv(orders_file, parse_dates=True)

    # Build prices df
    #  Get symbols from orders file
    symbols = orders["Symbol"].drop_duplicates().tolist()
    #  Get first and last date from order file
    start_date, end_date = orders["Date"].min(), orders["Date"].max()
    #  Get data from get_data function
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices.fillna(method="ffill").fillna(method="bfill")
    #  Add cash column that equals 1
    prices = prices.drop("SPY", axis=1)
    prices["CASH"] = 1
    # Build trades df
    #  Create df: column per symbol, across date range
    trades = pd.DataFrame(index=prices.index, columns=prices.columns, data=0)
    trades.iloc[0]["CASH"] = start_val
    #  Fill in trades by looping through orders file
    for i in range(orders.shape[0]):
        sym = orders.iloc[i]["Symbol"]
        order_type = orders.iloc[i]["Order"]
        shares = orders.iloc[i]["Shares"]
        date = orders.iloc[i]["Date"]
        trade_vol = shares if order_type == "BUY" else -shares
        trades.loc[date, sym] += trade_vol
        impact_adj_trade_vol = trade_vol * (1+impact) if order_type == "BUY" else trade_vol * (1-impact)
        trades.loc[date, "CASH"] -= (impact_adj_trade_vol * prices.loc[date, sym]) + commission

    # Build holdings df
    #  Cumulative sum of trades starting at starting_cash (start_val)
    holdings = trades.cumsum()
    # Build values df
    #  Values = prices * holdings
    values = prices * holdings
    # Sum values df to get portfolio value
    port_vals = values.sum(axis=1)

    return port_vals.to_frame()



def test_code():
    # this is a helper function you can use to test your code 			  		 			 	 	 		 		 	  		   	  			  	
    # note that during autograding his function will not be called. 			  		 			 	 	 		 		 	  		   	  			  	
    # Define input parameters 			  		 			 	 	 		 		 	  		   	  			  	

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders 			  		 			 	 	 		 		 	  		   	  			  	
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats 			  		 			 	 	 		 		 	  		   	  			  	
    # Here we just fake the data. you should use your code from previous assignments. 			  		 			 	 	 		 		 	  		   	  			  	
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2, 0.01, 0.02, 1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]

    # Compare portfolio against $SPX 			  		 			 	 	 		 		 	  		   	  			  	
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])


def author():
    return 'cfarr31'  # replace tb34 with your Georgia Tech username.


if __name__ == "__main__":
    test_code()
