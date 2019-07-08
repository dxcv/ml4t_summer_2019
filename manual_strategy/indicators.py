
"""Your code that implements your indicators as functions that operate on dataframes. The "main" code in
indicators.py should generate the charts that illustrate your indicators in the report.

Please add in an author function to each file.

 For your report, trade only the symbol JPM. This will enable us to more easily compare results.
You may use data from other symbols (such as SPY) to inform your strategy.
The in sample/development period is January 1, 2008 to December 31 2009.
The out of sample/testing period is January 1, 2010 to December 31 2011.
Starting cash is $100,000.
Allowable positions are: 1000 shares long, 1000 shares short, 0 shares.
Benchmark: The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM and holding that
position.
There is no limit on leverage.
Transaction costs for ManualStrategy: Commission: $9.95, Impact: 0.005.
Transaction costs for TheoreticallyOptimalStrategy: Commission: $0.00, Impact: 0.00.


https://www.sciencedirect.com/science/article/pii/S2405918815300179?via%3Dihub

"""
from util import get_data, plot_data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import os

# os.chdir(os.path.join(os.getcwd(), "manual_strategy"))

sym = ["JPM"]

in_start_date = pd.to_datetime("January 1, 2008")
in_end_date = pd.to_datetime("December 31 2009")
out_start_date = pd.to_datetime("January 1, 2010")
out_end_date = pd.to_datetime("December 31 2011")

# Trim to in sample range with a 30 day prior range for features
max_lookback = 30
start_date = in_start_date - datetime.timedelta(days=max_lookback)
end_date = in_end_date

# Normalize prices
prices = get_data(symbols=sym, dates=pd.date_range(start_date, end_date), addSPY=False)
prices = prices.fillna(method="ffill").fillna(method="bfill")
prices["JPM"] = prices["JPM"] / prices["JPM"][0]

plot_start = in_start_date + datetime.timedelta(days=180)
plot_end = in_start_date + datetime.timedelta(days=250)


""" Moving Average """

# Simple statistic mean of previous n day closing price:


moving_average_df = prices.copy()
moving_average_df["moving_average"] = moving_average_df[sym].rolling(window=12).mean()
moving_average_df["moving_std"] = moving_average_df[sym].rolling(window=12).std()
moving_average_df["simple_moving_average"] = moving_average_df.apply(lambda x: (x[sym[0]] / x["moving_average"]) - 1, axis=1)

# Add bollinger bands
moving_average_df["bollinger_max"] = moving_average_df["moving_average"] + moving_average_df["moving_std"] * 2
moving_average_df["bollinger_min"] = moving_average_df["moving_average"] - moving_average_df["moving_std"] * 2

# Bollinger bands as indicator
# (price - sma) / (2 * std)
moving_average_df["bollinger_band"] = moving_average_df.apply(
    lambda x: (x["JPM"] - x["moving_average"]) / (2 * x["moving_std"]), axis=1)


# Plot
plot_df = moving_average_df.loc[plot_start:plot_end, ["JPM", "moving_average", "bollinger_max", "bollinger_min"]].copy()

plot_df.plot(title="Simple Moving Average\nBollinger Bands")


def above_upper(i, df):
    return df.loc[i, "JPM"] > df.loc[i, "bollinger_max"]


def below_lower(i, df):
    return df.loc[i, "JPM"] < df.loc[i, "bollinger_min"]


for i in range(1, len(plot_df.index)):

    c_i = plot_df.index[i]
    l_i = plot_df.index[i-1]

    if above_upper(l_i, plot_df) and not above_upper(c_i, plot_df):
        plt.axvline(x=c_i, color="red")
    if below_lower(l_i, plot_df) and not below_lower(c_i, plot_df):
        plt.axvline(x=c_i, color="green")

plt.show()

# TODO Indicator signal: Price crosses over sma, along with momentum for signal
#  If SMA is positive, sell, if negative, buy: given that momentum is strong when it crosses

# moving_average_df.loc[plot_start:plot_end, :].plot()
# plt.show()

# TODO Indicator signal: Price crosses back into bands


""" Moving Average Convergence and Divergence """

# TODO Learn to interpret
#  When the shorter-term MA cross above the longer-term MA, it's a buy signal
#  When the shorter-term MA crosses below the longer-term MA, its a sell signal
# https://towardsdatascience.com/implementing-macd-in-python-cc9b2280126a
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html

# Shows the relationship between two exponential moving averages of prices

# MACD = EMA(12) - EMA(26)
# EMA(i) = (CP(i) - EMA(i-1)) * Multiplier + EMA(i-1)
# Multiplier = 2 / (no of days to be considered + 1)

macd_df = prices.copy()

macd_df["ema_12"] = macd_df[sym].ewm(span=12, adjust=False).mean()
macd_df["ema_26"] = macd_df[sym].ewm(span=26, adjust=False).mean()
macd_df["macd"] = macd_df["ema_12"] - macd_df["ema_26"]

# Signal or average series
macd_df["macd_signal_line"] = macd_df["macd"].ewm(span=9, adjust=False).mean()

# Divergence line
macd_df["divergence"] = (macd_df["macd"] / macd_df["macd_signal_line"]) - 1

macd_df = macd_df.loc[in_start_date:in_end_date, :].copy()

# Plot the price and macd/macd signal line
# Add lines where they cross, color green if bullish cross, red is bearish cross
macd_df["bullish_macd"] = macd_df.apply(lambda x: x["macd"] > x["macd_signal_line"], axis=1)
macd_df["bullish_cross_over"] = macd_df["bullish_macd"].rolling(window=2).apply(lambda x: all(x[-1:]) and not x[0]).fillna(0).astype(bool)
macd_df["bearish_cross_over"] = macd_df["bullish_macd"].rolling(window=2).apply(lambda x: not any(x[-1:]) and x[0]).fillna(0).astype(bool)




# plot_start = in_start_date + datetime.timedelta(days=60)
# plot_end = in_start_date + datetime.timedelta(days=180)
plot_cols = ["JPM", "macd", "macd_signal_line"]
line_cols = ["bullish_cross_over", "bearish_cross_over"]
plot_df = macd_df.loc[plot_start:plot_end, plot_cols + line_cols]

plot_df.loc[:, plot_cols].plot(secondary_y="JPM", title="Moving Average Convergence/Divergence")
for i in plot_df.index:
    if plot_df.loc[i, "bullish_cross_over"]:
        plt.axvline(x=i, color="green")
    if plot_df.loc[i, "bearish_cross_over"]:
        plt.axvline(x=i, color="red")

plt.show()

# the MACD series proper, the "signal" or "average" series, and the "divergence" series which is the difference
# between the two. The MACD series is the difference between a "fast" (short period) exponential moving average (
# EMA), and a "slow" (longer period) EMA of the price series. The average series is an EMA of the MACD series itself.

# TODO How to use:

# https://en.wikipedia.org/wiki/MACD

# A "signal-line crossover" occurs when the MACD and signal lines cross; that is, when the divergence (the bar
# graph) changes sign. The standard interpretation of such an event is a recommendation to buy if the MACD line
# crosses up through the average line (a "bullish" crossover), or to sell if it crosses down through the average line
# (a "bearish" crossover).[6] These events are taken as indications that the trend in the stock is about to
# accelerate in the direction of the crossover.

# Zero crossover A "zero crossover" event occurs when the MACD series changes sign, that is, the MACD line crosses
# the horizontal zero axis. This happens when there is no difference between the fast and slow EMAs of the price
# series. A change from positive to negative MACD is interpreted as "bearish", and from negative to positive as
# "bullish". Zero crossovers provide evidence of a change in the direction of a trend but less confirmation of its
# momentum than a signal line crossover.

# Divergence A "positive divergence" or "bullish divergence" occurs when the price makes a new low but the MACD does
# not confirm with a new low of its own. A "negative divergence" or "bearish divergence" occurs when the price makes
# a new high but the MACD does not confirm with a new high of its own.[7] A divergence with respect to price may
# occur on the MACD line and/or the MACD Histogram

# My strategy:
#  If macd is positive (bullish) and macd > signal_line (signal-line cross-over) buy
#  If macd is negative (bearish) and macd > signal_line, hold
#  If macd is positive and macd < signal line, hold
#  If macd is negative and macd < signal line, sell


# View a shorter period to view the indicator lines


"""
MOMENTUM INDICATORS
"""

""" Lecture version Momentum """

momentum_df = prices.copy()
momentum_df["momentum"] = momentum_df["JPM"].rolling(window=9).apply(lambda x: (x[-1] / x[0]) - 1)

# plot_start = in_start_date + datetime.timedelta(days=60)
# plot_end = in_start_date + datetime.timedelta(days=180)

plot_df = momentum_df.loc[plot_start:plot_end, :]

plots = plot_df.plot(secondary_y="momentum", title="Momentum")

plt.axhline(y=.1, color="green")

plt.axhline(y=-.1, color="red")
plt.show()

# Short strategy: sell signal from macd with momentum? TODO Start here!


""" Stochastic D """

# Stochastic provides a mean of measuring price movement velocity. K% measures the relative position of current
# closing price in a certain time range, wheas D% specific the three day moving average of K%. K%(i) = (cp(i) - Lt) /
# (Ht - Lt) * 100 D%(i) = (K%(i-2) + K%(i-1) + K%(i)) / 3 Where cp(i) is the closing price, Lt is the lowest price of
# the last t days, Ht is the highest price of last t days. https://www.investopedia.com/articles/technical/073001.asp

# The K line is faster than the D line the slower of the two. The investor needs to watch as the D line and the price
# of the issue begin to change and move into either the overbought (over the 80 line) or the oversold (under the 20
# line) positions. The investor needs to consider selling the stock when the indicator moves above the 80 level.
# Conversely, the investor needs to consider buying an issue that is below the 20 line and is starting to move up
# with increased volume.

stoch_kd_df = prices.copy()

stoch_kd_df["K"] = stoch_kd_df[sym[0]].rolling(window=9).apply(
    lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)))
stoch_kd_df["D"] = stoch_kd_df["K"].rolling(window=3).mean()

# plot_start = in_start_date + datetime.timedelta(days=60)
# plot_end = in_start_date + datetime.timedelta(days=180)

plot_df = stoch_kd_df.loc[plot_start:plot_end, ["JPM", "D"]]

plot_df.plot(secondary_y=["D"], title="Stochastic D")

plt.axhline(y=.80, color="green")

plt.axhline(y=.20, color="red")

plt.show()

# When D is above the 80% mark, the stock is in overbought territory, consider selling
# When D is below the 20% mark, the stock is in oversold territory, consider buying


# TODO Produce a single DF with all indicators (no helper data)

# sma_indicator
bollinger_indicator = moving_average_df.loc[:, "bollinger_band"].copy()
# macd, signal_line
macd = macd_df.loc[:, ["bearish_cross_over"]].copy()
# momentum
momentum = momentum_df.loc[:, ["momentum"]].copy()
# Stochastic KD
stochastic_d = stoch_kd_df.loc[:, ["D"]].copy()


# Combine all into indicators df and filter to in_sample range
indicator_df = pd.concat([bollinger_indicator, macd, momentum, stochastic_d], axis=1)
indicator_df = indicator_df.loc[in_start_date:in_end_date].copy()


# Strategy
# Short selling
# Bearish macd cross-over after recent overbought indicators from stochastic D and Momentum
#  Any overbought in last 3 days (D > .8 or Momentum > .1)

# Longing
# Bollinger indicator falls below -1

