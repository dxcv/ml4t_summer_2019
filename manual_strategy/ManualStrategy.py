"""Code implementing a ManualStrategy object (your manual strategy). It should implement testPolicy() which returns a
trades data frame (see below). The main part of this code should call marketsimcode as necessary to generate the
plots used in the report.

Please add in an author function to each file.




TODO's
 Setup functions in indicators that work with data frames
 Add main to indicators that generates all plots (and saves them)
 Implement ManualStrategy object. Add function testPolicy
 Add main to ManualStrategy.py to generate plots and save them
 Implement TheoreticallyOptimalStrategy object. Add function testPolicy
 Add main to TheoreticallyOptimalStrategy.py to generate and save plots







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


def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000, return_signal=False):
    # Trim to in sample range with a 30 day prior range for features
    max_lookback = 30
    start_date = sd - datetime.timedelta(days=max_lookback)
    end_date = ed

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
    bearish_macd = indicator_df.loc[:, "bearish_cross_over"]
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

    # Compare to optimal
    # optimal = prices.rolling(window=2).apply(lambda x: -1 if x[-1] < x[0] else 1 if x[0] < x[1] else 0).shift(
    #         -1).fillna(0).astype(int)
    #
    # optimal = pd.merge(signal_df, optimal, left_index=True, right_index=True)

    # signal_df = signal_df.apply(lambda x: x["macd_indicator"] if not x["sma_indicator"] else x["sma_indicator"], axis=1).to_frame()
    # signal_df.columns = ["signal"]

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

    return trades, signal_df


def author():
    return 'cfarr31'  # replace tb34 with your Georgia Tech username.

# TODO Use indicators_df to create a buy/sell/hold signal column named "signal"

# My strategy:
# An example of a price filter would be to buy if the MACD line breaks above the signal line and then remains above
# it for three days
#  If macd is positive (bullish)
#  If macd is negative (bearish)
#  If macd > signal_line (signal-line cross-over) buy
#  If macd < signal line, sell

# Iteratively improve strategy adding one indicator at a time...





symbol = "JPM"

in_start_date = pd.to_datetime("January 1, 2008")
in_end_date = pd.to_datetime("December 31 2009")
out_start_date = pd.to_datetime("January 1, 2010")
out_end_date = pd.to_datetime("December 31 2011")

sd = out_start_date
ed = out_end_date


# Benchmark portvals
prices = get_data(symbols=["JPM"], dates=pd.date_range(sd, ed), addSPY=False)
prices = prices.dropna()  # Instead of fill when creating trades
benchmark_trades = pd.DataFrame(columns=["JPM"], index=prices.index, data=0)
benchmark_trades.iloc[0] = 1000
benchmark_portvals = compute_portvals(benchmark_trades, commission=9.95, impact=0.005)
benchmark_portvals.columns = ["Benchmark"]

# Optimal portvals
portvals = compute_portvals(trades, commission=9.95, impact=0.005)
portvals.columns = ["Manual"]

plot_df = pd.concat([portvals, benchmark_portvals], axis=1)

plot_df = plot_df / plot_df.iloc[0]
plot_df.loc[:, ["Benchmark", "Manual"]].plot(color=["green", "red"], title="Manual Strategy", secondary_y="JPM")

for i in signal_df.index:
    if i not in plot_df.index:
        continue
    y = plot_df.loc[i, "Benchmark"]
    if signal_df.loc[i, "signal"] == -1:
        plt.vlines(x=i, ymin=y-.01, ymax=y, color="black")
    elif signal_df.loc[i, "signal"] == 1:
        plt.vlines(x=i, ymin=y, ymax=y+.01, color="lightblue")
plt.show()

print "Performance %.1f%%" % ((plot_df["Manual"][-1] - 1) * 100)

