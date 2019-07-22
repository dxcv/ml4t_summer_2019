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
from QLearner import QLearner
from datetime import timedelta
from indicators import add_all_indicators
import itertools
import numpy as np


# import os
# os.chdir(os.path.join(os.getcwd(), "strategy_learner"))


class StrategyLearner(object):

    # constructor 			  		 			 	 	 		 		 	  		   	  			  	
    def __init__(self, verbose=False, impact=0.0, dyna=200, epochs=3):
        self.verbose = verbose
        self.impact = impact
        self._z_params = dict()
        self._state_dict = dict()
        self._learner = None
        self._state_order = []
        self._actions = [-1, 0, 1]  # long 1, cash 0, short -1
        self._dyna = dyna
        self._epochs = epochs

    # this method should create a QLearner, and train it for trading 			  		 			 	 	 		 		 	  		   	  			  	
    def addEvidence(self, symbol="IBM",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 1, 1),
                    sv=10000, n=1):
        # add your code to do learning here

        # Get price info
        syms = [symbol]
        adj_sd = sd - timedelta(days=30)
        dates = pd.date_range(adj_sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        # Calculate indicators
        df = add_all_indicators(prices, syms[0], add_helper_data=False)

        # Filter to time range
        df = df.loc[sd:ed, :].copy()

        # Get state df
        state_df = self.fit_transform_state(df, symbol, n_days=n)

        # Initialize Q-learner
        self._learner = QLearner(num_states=len(self._state_dict), num_actions=len(self._actions), dyna=self._dyna)

        # Fit learner
        for _ in range(self._epochs):
            self.fit_learner(state_df)
        return

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="IBM",
                   sd=dt.datetime(2009, 1, 1),
                   ed=dt.datetime(2010, 1, 1),
                   sv=10000):

        # Get price info
        syms = [symbol]
        adj_sd = sd - timedelta(days=30)
        dates = pd.date_range(adj_sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        # Calculate indicators
        df = add_all_indicators(prices, syms[0], add_helper_data=False)

        # Filter to time range
        df = df.loc[sd:ed, :].copy()

        # Get state df
        state_df = self.transform_state(df)

        trades_df = self.test_learner(state_df, symbol)

        return trades_df

    def fit_learner(self, state_df):
        # Single iteration through training data
        # Mix the order up so dyna doesn't overfit to early experiences
        # return nothing
        i = 0
        state = list(state_df.iloc[i].loc[self._state_order])
        position = self._actions[1]  # Action cash is position 0 at index 1
        state.append(position)
        state_num = self._state_dict[tuple(state)]
        action = self._learner.querysetstate(state_num)

        for i in range(1, state_df.shape[0]):
            orig_position = position
            # Calculate reward
            position = self._actions[action]
            reward = state_df.iloc[i]["reward"] * position

            # Adjust reward for impact
            # Depending on original position...
            # If the position changes, reduce reward by impact
            if orig_position != position:
                reward = reward - (abs(reward) * self.impact)

            # Calculate reward
            # Long (1) and price goes down...
            # return * 1
            # Long (1) and price goes up...
            # return * 1
            # Cash (0) and price goes down...
            # return * -1
            # Cash (0) and price goes up...
            # return * -1
            # Short (-1) and price goes down...
            # return * -1
            # Short (-1) and price goes up...
            # return * -1

            # Update state
            state = list(state_df.loc[:, self._state_order].iloc[i])
            state.append(position)
            state_num = self._state_dict[tuple(state)]
            # Query with new state and reward for last action
            action = self._learner.query(state_num, reward)

        return

    def test_learner(self, state_df, symbol):

        # Test Q-Learner in sample for cumulative return (create trades df)
        actions_list = []

        i = 0
        state = list(state_df.iloc[i].loc[self._state_order])
        position = self._actions[1]  # Action cash is position 0 at index 1
        state.append(position)
        state_num = self._state_dict[tuple(state)]
        action = self._learner.querysetstate(state_num)
        actions_list.append(self._actions[action])

        for i in range(state_df.shape[0]):
            position = self._actions[action]
            # Update state
            state = list(state_df.loc[:, self._state_order].iloc[i])
            state.append(position)
            state_num = self._state_dict[tuple(state)]
            action = self._learner.querysetstate(state_num)
            actions_list.append(self._actions[action])

        trades_df = pd.DataFrame(index=state_df.index, data=actions_list[:-1], columns=["action"])

        trades_df[symbol] = 0

        current_position = 0

        for i in trades_df.index:

            action = trades_df.loc[i, "action"]
            if action == current_position:
                trades_df.loc[i, symbol] = 0
            elif action == -1 and current_position == 1:
                trades_df.loc[i, symbol] = -2000
            elif action == 1 and current_position == -1:
                trades_df.loc[i, symbol] = 2000
            elif action == -1 and current_position == 0:
                trades_df.loc[i, symbol] = -1000
            elif action == 1 and current_position == 0:
                trades_df.loc[i, symbol] = 1000
            elif action == 0 and current_position == 1:
                trades_df.loc[i, symbol] = -1000
            elif action == 0 and current_position == -1:
                trades_df.loc[i, symbol] = 1000
            else:
                raise ValueError("Impossible")

            current_position = action

        return trades_df.drop("action", axis=1)

    def fit_transform_state(self, df, symbol, n_days):
        # Symbol specifies the price column name
        # Accepts df with price and indicators
        # Digitize indicators
        # Store z parameters for later use
        # Calculate reward
        # Create state dict for mapping digitized state to integer
        # Return state_df

        # Create state DF
        state_df = pd.DataFrame(index=df.index)

        # Add and digitize indicators

        # Calculate target: n-day future return
        state_df["reward"] = df[symbol].iloc[::-1].rolling(window=n_days + 1).apply(lambda x: (x[0] / x[-1]) - 1).iloc[::-1]

        # Impact adjusted reward
        state_df["buy_impact_reward"]

        state_values = []

        # How many states are there, and how do I map them to a single integer?
        # unique_boll = range(state_df["bollinger_band"].max() + 1)

        # Add bollinger band
        state_df["bollinger_band"], unique_n = self.digitize_bollinger(df["bollinger_band"])
        unique_boll = list(range(unique_n))
        state_values.append(unique_boll)
        self._state_order.append("bollinger_band")

        # Add divergence
        state_df["divergence"], unique_n = self.digitize_divergence(df["divergence"])
        unique_div = list(range(unique_n))
        state_values.append(unique_div)
        self._state_order.append("divergence")

        # Add momentum
        state_df["momentum"], unique_n = self.digitize_momentum(df["momentum"])
        unique_mom = list(range(unique_n))
        state_values.append(unique_mom)
        self._state_order.append("momentum")

        # Add D
        state_df["D"], unique_n = self.digitize_d(df["D"])
        unique_d = list(range(unique_n))
        state_values.append(unique_d)
        self._state_order.append("D")

        state_values.append(self._actions)

        # Drop NA's, without reward
        state_df = state_df.dropna()

        all_states = list(itertools.product(*state_values))
        self._state_dict = {k: v for v, k in enumerate(all_states)}
        return state_df

    def transform_state(self, df):
        # Accepts df with indicators
        # Digitize indicators using existing params
        # Return state_df

        # Create state DF
        state_df = pd.DataFrame(index=df.index)

        state_df["bollinger_band"], _ = self.digitize_bollinger(df["bollinger_band"])
        state_df["divergence"], _ = self.digitize_divergence(df["divergence"])
        state_df["momentum"], _ = self.digitize_momentum(df["momentum"])
        state_df["D"], _ = self.digitize_d(df["D"])

        return state_df

    @staticmethod
    def digitize_bollinger(indicator):

        bins = [-1.03, 0, 1.03]

        return np.digitize(indicator, bins), len(bins) + 1

    @staticmethod
    def digitize_divergence(indicator):

        bins = [-.25, 0, .25]

        return np.digitize(indicator, bins), len(bins) + 1

    @staticmethod
    def digitize_momentum(indicator):

        bins = [-.27, .27]

        return np.digitize(indicator, bins), len(bins) + 1

    @staticmethod
    def digitize_d(indicator):
        bins = [.2, .8]
        return np.digitize(indicator, bins), len(bins) + 1

    def author(self):
        return 'cfarr31'
