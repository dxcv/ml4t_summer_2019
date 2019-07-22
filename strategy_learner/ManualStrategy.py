"""Code implementing a ManualStrategy object (your manual strategy). It should implement testPolicy() which returns a
trades data frame (see below). The main part of this code should call marketsimcode as necessary to generate the
plots used in the report.

Please add in an author function to each file.
"""

from util import get_data, plot_data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import os
import datetime as dt
from marketsimcode import compute_portvals
from indicators import add_all_indicators


class ManualStrategy:

    @staticmethod
    def calculate_statistics(port_val):
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
        return {"cum_ret": cr, "avg_day_ret": adr, "std_day_ret": sddr}

    @staticmethod
    def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000, return_signal=False):
        # Trim to in sample range with a 30 day prior range for features
        max_lookback = 30
        start_date = pd.to_datetime(sd) - datetime.timedelta(days=max_lookback)
        end_date = pd.to_datetime(ed)

        # Get prices
        prices = get_data(symbols=[symbol], dates=pd.date_range(start_date, end_date), addSPY=False)
        prices = prices.fillna(method="ffill").fillna(method="bfill")

        # Add indicators
        indicator_df = add_all_indicators(prices, "JPM", False)

        # Filter to range
        indicator_df = indicator_df.loc[sd:ed, :].copy()

        # Strategy
        # Short selling
        # Bearish macd cross-over after recent overbought indicators from stochastic D and Momentum
        bearish_macd = indicator_df.loc[:, "divergence"].rolling(window=2).apply(lambda x: not any(x[-1:] < 0) and x[0] < 0).fillna(0).astype(bool)

        #  Any overbought in last 3 days (D > .8 or Momentum > .1)
        overbought_momentum = indicator_df["momentum"].rolling(window=12).apply(lambda x: any(x > .1)).fillna(0).astype(
            bool)
        overbought_stochastic_d = indicator_df["D"].rolling(window=12).apply(lambda x: any(x > .8)).fillna(0).astype(bool)
        overbought = overbought_momentum | overbought_stochastic_d
        short_sell = (bearish_macd & overbought).to_frame()
        short_sell.columns = ["short"]
        short_sell.sum()

        # Longing
        # Bollinger indicator falls below -1
        long_buy = indicator_df["bollinger_band"].apply(lambda x: x < -1.03).to_frame()
        long_buy.columns = ["long"]
        long_buy.sum()

        signal_df = pd.concat([short_sell, long_buy], axis=1)
        signal_df = signal_df.apply(lambda x: 1 if x["long"] else -1 if x["short"] else 0, axis=1).to_frame()
        signal_df.columns = ["signal"]

        # Clean up the signal df
        current_position = 0
        for i in signal_df.index:
            if signal_df.loc[i, "signal"] == current_position:
                signal_df.loc[i, "signal"] = 0
            if signal_df.loc[i, "signal"] != 0:
                current_position = signal_df.loc[i, "signal"]

        symbol = "JPM"
        prices = get_data(symbols=["JPM"], dates=pd.date_range(sd, ed), addSPY=False)
        prices = prices.dropna()  # Instead of fill when creating trades

        prices = pd.merge(prices, signal_df, how="left", left_index=True, right_index=True)

        trades = pd.DataFrame(columns=[symbol], index=prices.index, data=0)

        current_position = 0

        for i in prices.index:

            signal = prices.loc[i, "signal"]

            if signal == -1 and current_position == 1:
                # order = "SELL"
                trades.loc[i, symbol] = -2000
                current_position = -1
            elif signal == 1 and current_position == -1:
                # order = "BUY"
                trades.loc[i, symbol] = 2000
                current_position = 1
            elif signal == -1 and current_position == 0:
                trades.loc[i, symbol] = -1000
                current_position = -1
            elif signal == 1 and current_position == 0:
                trades.loc[i, symbol] = 1000
                current_position = 1
            else:
                trades.loc[i, symbol] = 0

        return trades

    @staticmethod
    def author():
        return 'cfarr31'  # replace tb34 with your Georgia Tech username.
