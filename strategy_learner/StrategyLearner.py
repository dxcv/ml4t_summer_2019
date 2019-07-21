""" 			  		 			 	 	 		 		 	  		   	  			  	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
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

import datetime as dt
import pandas as pd
import util as ut
import random
from .QLearner import QLearner


# import os
# os.chdir(os.path.join(os.getcwd(), "strategy_learner"))


class StrategyLearner(object):

    # constructor 			  		 			 	 	 		 		 	  		   	  			  	
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

    # this method should create a QLearner, and train it for trading 			  		 			 	 	 		 		 	  		   	  			  	
    def addEvidence(self, symbol="IBM", \
                    sd=dt.datetime(2008, 1, 1), \
                    ed=dt.datetime(2009, 1, 1), \
                    sv=10000):
        # add your code to do learning here
        # Get price info
        syms = [symbol]
        dates = pd.date_range(adj_sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        # Calculate indicators
        df = add_all_indicators(prices, syms[0], add_helper_data=False)
        # Filter to time range
        df = df.loc[sd:ed, :].copy()

        pass
        # # example usage of the old backward compatible util function
        # syms = [symbol]
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        # prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print prices
        #
        # # example use with new colname
        # volume_all = ut.get_data(syms, dates,
        #                          colname="Volume")  # automatically adds SPY
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print volume

    # this method should use the existing policy and test it against new data 			  		 			 	 	 		 		 	  		   	  			  	
    def testPolicy(self, symbol="IBM", \
                   sd=dt.datetime(2009, 1, 1), \
                   ed=dt.datetime(2010, 1, 1), \
                   sv=10000):

        # here we build a fake set of trades 			  		 			 	 	 		 		 	  		   	  			  	
        # your code should return the same sort of data 			  		 			 	 	 		 		 	  		   	  			  	
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY 			  		 			 	 	 		 		 	  		   	  			  	
        trades = prices_all[[symbol, ]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later 			  		 			 	 	 		 		 	  		   	  			  	
        trades.values[:, :] = 0  # set them all to nothing
        trades.values[0, :] = 1000  # add a BUY at the start
        trades.values[40, :] = -1000  # add a SELL
        trades.values[41, :] = 1000  # add a BUY
        trades.values[60, :] = -2000  # go short from long
        trades.values[61, :] = 2000  # go long from short
        trades.values[-1, :] = -1000  # exit on the last day
        if self.verbose: print type(trades)  # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades


# Create simulation for development
from datetime import timedelta
from .indicators import *
import itertools


symbol = "IBM"
sd = dt.datetime(2008, 1, 1)
adj_sd = sd - timedelta(days=30)
ed = dt.datetime(2009, 1, 1)
sv = 10000

# Get price info
syms = [symbol]
dates = pd.date_range(adj_sd, ed)
prices_all = ut.get_data(syms, dates)  # automatically adds SPY
prices = prices_all[syms]  # only portfolio symbols
prices_SPY = prices_all['SPY']  # only SPY, for comparison later
# Calculate indicators
df = add_all_indicators(prices, syms[0], add_helper_data=False)
# Filter to time range
df = df.loc[sd:ed, :].copy()

# Create new df
state_df = pd.DataFrame(index=df.index)
# Discretize indicators
# TODO There needs to be some lag versions added to describe the direction
# Find ranges, divide into.... 3?
# Ranges should split right at 0 if there is a crossover
# pos_range = df["bollinger_band"].max() + .01
# neg_range = df["bollinger_band"].min()
# bins = np.array([neg_range / 2, 0, pos_range / 2])
# state_df["bollinger_band"] = np.digitize(df["bollinger_band"], bins)
#
#
# pos_range = df["divergence"].max() + .01
# neg_range = df["divergence"].min()
# bins = np.array([neg_range / 2, 0, pos_range / 2])
# state_df["divergence"] = np.digitize(df["divergence"], bins)

range_max = df["D"].max() + .01
range_min = df["D"].min()
bins = np.arange(range_min, range_max, 1. / 3.)
state_df["D"] = np.digitize(df["D"], bins)


# Convert each to z-score prior to digitizing
def z_digitize(indicator, mu=None, std=None):
    if mu is None:
        mu = indicator.mean()
    if std is None:
        std = indicator.std()
    z = (indicator - mu) / std
    bins = np.array([-1, -.1, 0, .1, 1])
    return np.digitize(z, bins), mu, std


state_df["bollinger_band"], boll_mu, boll_std = z_digitize(df["bollinger_band"])
state_df["divergence"], div_mu, div_std = z_digitize(df["divergence"])
state_df["momentum"], momentum_mu, momentum_std = z_digitize(df["momentum"])

# Calculate target: n-day future return
n=1
state_df["reward"] = df[symbol].iloc[::-1].rolling(window=n+1).apply(lambda x: (x[0] / x[-1]) - 1).iloc[::-1]

# Drop NA's, without reward
state_df = state_df.dropna()

# How many states are there, and how do I map them to a single integer?
# Bollinger, divergence, and momentum: 6 each
unique_boll = range(state_df["bollinger_band"].max() + 1)
unique_div = range(state_df["divergence"].max() + 1)
unique_mom = range(state_df["momentum"].max() + 1)
# D: 4
unique_d = range(1, state_df["D"].max() + 1)
unique_actions = [-1, 0, 1]  # long 1, cash 0, short -1


state_values = [unique_boll, unique_div, unique_mom, unique_d, unique_actions]
state_order = ["bollinger_band", "divergence", "momentum", "D"]
all_states = list(itertools.product(*state_values))
state_dict = {k: v for v, k in enumerate(all_states)}

# Initialize Q-learner
q_learner = QLearner(num_states=len(state_dict), num_actions=3, dyna=200)
# Loop through data and train Q-learner

i = 0
state = list(state_df.iloc[i].loc[state_order])
position = unique_actions[1]  # Action cash is position 0 at index 1
state.append(position)
state_num = state_dict[tuple(state)]
action = q_learner.querysetstate(state_num)

for i in range(1, state_df.shape[0]):

    # Calculate reward
    position = unique_actions[action]
    reward = state_df.iloc[i]["reward"] * position

    # Update state
    state = list(state_df.loc[:, state_order].iloc[i])
    state.append(position)
    state_num = state_dict[tuple(state)]
    # Query with new state and reward for last action
    action = q_learner.query(state_num, reward)


# Test Q-Learner in sample for cumulative return (create trades df)
actions_list = []

i = 0
state = list(state_df.iloc[i].loc[state_order])
position = unique_actions[1]  # Action cash is position 0 at index 1
state.append(position)
state_num = state_dict[tuple(state)]
action = q_learner.querysetstate(state_num)
actions_list.append(unique_actions[action])
for i in range(state_df.shape[0]):
    position = unique_actions[action]
    # Update state
    state = list(state_df.loc[:, state_order].iloc[i])
    state.append(position)
    state_num = state_dict[tuple(state)]
    action = q_learner.querysetstate(state_num)
    actions_list.append(unique_actions[action])

trades_df = pd.DataFrame(index=state_df.index, data=actions_list[:-1], columns=["action"])

trades_df[symbol] = 0

current_position = 0

for i in trades_df.index:

    action = trades_df.loc[i, "action"]
    if action == current_position:
        trades_df.loc[i, symbol] = 0
    elif action == -1 and current_position == 1:
        trades_df.loc[i, symbol] = -2000
        current_position = -1
    elif action == 1 and current_position == -1:
        trades_df.loc[i, symbol] = 2000
        current_position = 1
    elif action == -1 and current_position == 0:
        trades_df.loc[i, symbol] = -1000
        current_position = -1
    elif action == 1 and current_position == 0:
        trades_df.loc[i, symbol] = 1000
        current_position = 1
    elif action == 0 and current_position == 1:
        trades_df.loc[i, symbol] = -1000
    elif action == 0 and current_position == -1:
        trades_df.loc[i, symbol] = 1000
    else:
        raise ValueError("Impossible")
    current_position = action


# Calculate return

from marketsimcode import compute_portvals

port_vals = compute_portvals("IBM", trades_df)

# Calculate statistics
cum_return = port_vals.iloc[-1] / port_vals.iloc[0] - 1
