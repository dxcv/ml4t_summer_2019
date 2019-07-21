
"""Your code that implements your indicators as functions that operate on dataframes. The "main" code in
indicators.py should generate the charts that illustrate your indicators in the report.

Please add in an author function to each file.

"""
from util import get_data, plot_data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import os

""" Bollinger Band """

# Simple statistic mean of previous n day closing price:


def add_bollinger_band_indicator(df, symbol="JPM", add_helper_data=False):
    bollinger_df = df.copy()
    bollinger_df["moving_average"] = bollinger_df[symbol].rolling(window=12).mean()
    bollinger_df["moving_std"] = bollinger_df[symbol].rolling(window=12).std()
    bollinger_df["simple_moving_average"] = bollinger_df.apply(lambda x: (x[symbol] / x["moving_average"]) - 1, axis=1)

    # Add bollinger bands
    bollinger_df["bollinger_max"] = bollinger_df["moving_average"] + bollinger_df["moving_std"] * 2
    bollinger_df["bollinger_min"] = bollinger_df["moving_average"] - bollinger_df["moving_std"] * 2

    # Bollinger bands as indicator
    # (price - sma) / (2 * std)
    bollinger_df["bollinger_band"] = bollinger_df.apply(
        lambda x: (x[symbol] - x["moving_average"]) / (2 * x["moving_std"]), axis=1)

    if add_helper_data:
        cols = [symbol, "moving_average", "bollinger_max", "bollinger_min", "bollinger_band"]
    else:
        cols = ["bollinger_band"]

    return bollinger_df.loc[:, cols]


""" Moving Average Convergence and Divergence """

#  When the shorter-term MA cross above the longer-term MA, it's a buy signal
#  When the shorter-term MA crosses below the longer-term MA, its a sell signal
# https://towardsdatascience.com/implementing-macd-in-python-cc9b2280126a
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html

# Shows the relationship between two exponential moving averages of prices

# MACD = EMA(12) - EMA(26)
# EMA(i) = (CP(i) - EMA(i-1)) * Multiplier + EMA(i-1)
# Multiplier = 2 / (no of days to be considered + 1)


def add_macd(df, symbol, add_helper_data=False):
    macd_df = df.copy()

    macd_df["ema_12"] = macd_df[symbol].ewm(span=12, adjust=False).mean()
    macd_df["ema_26"] = macd_df[symbol].ewm(span=26, adjust=False).mean()
    macd_df["macd"] = macd_df["ema_12"] - macd_df["ema_26"]

    # Signal or average series
    macd_df["macd_signal_line"] = macd_df["macd"].ewm(span=9, adjust=False).mean()

    # Divergence line
    macd_df["divergence"] = (macd_df["macd"] / macd_df["macd_signal_line"]) - 1

    # Plot the price and macd/macd signal line
    # Add lines where they cross, color green if bullish cross, red is bearish cross
    macd_df["bullish_macd"] = macd_df.apply(lambda x: x["macd"] > x["macd_signal_line"], axis=1)
    macd_df["bullish_cross_over"] = macd_df["bullish_macd"].rolling(window=2).apply(lambda x: all(x[-1:]) and not x[0]).fillna(0).astype(bool)
    macd_df["bearish_cross_over"] = macd_df["bullish_macd"].rolling(window=2).apply(lambda x: not any(x[-1:]) and x[0]).fillna(0).astype(bool)

    if add_helper_data:
        cols = [symbol, "macd", "macd_signal_line", "bullish_macd", "bullish_cross_over", "bearish_cross_over"]
    else:
        cols = ["divergence"]
    return macd_df.loc[:, cols]


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


"""
MOMENTUM INDICATORS
"""

""" Lecture version Momentum """


def add_momentum(df, symbol, add_helper_data=False):
    momentum_df = df.copy()
    momentum_df["momentum"] = momentum_df[symbol].rolling(window=9).apply(lambda x: (x[-1] / x[0]) - 1)

    if add_helper_data:
        cols = [symbol, "momentum"]
    else:
        cols = ["momentum"]

    return momentum_df.loc[:, cols]

# plot_start = in_start_date + datetime.timedelta(days=60)
# plot_end = in_start_date + datetime.timedelta(days=180)

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


def add_stochastic_d(df, symbol, add_helper_data=False):
    stoch_kd_df = df.copy()

    stoch_kd_df["K"] = stoch_kd_df[symbol].rolling(window=9).apply(
        lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)))
    stoch_kd_df["D"] = stoch_kd_df["K"].rolling(window=3).mean()

    if add_helper_data:
        cols = [symbol, "K", "D"]
    else:
        cols = ["D"]

    return stoch_kd_df.loc[:, cols]


# When D is above the 80% mark, the stock is in overbought territory, consider selling
# When D is below the 20% mark, the stock is in oversold territory, consider buying

# Strategy
# Short selling
# Bearish macd cross-over after recent overbought indicators from stochastic D and Momentum
#  Any overbought in last 3 days (D > .8 or Momentum > .1)

# Longing
# Bollinger indicator falls below -1


def add_all_indicators(df, symbol, add_helper_data=False):
    bollinger = add_bollinger_band_indicator(df, symbol, add_helper_data)
    macd = add_macd(df, symbol, add_helper_data)
    momentum = add_momentum(df, symbol, add_helper_data)
    stochastic_d = add_stochastic_d(df, symbol, add_helper_data)
    df = pd.concat([df, bollinger, macd, momentum, stochastic_d], axis=1)
    return df


def author():
    return 'cfarr31'  # replace tb34 with your Georgia Tech username.


if __name__ == "__main__":

    symbol = "JPM"

    in_start_date = pd.to_datetime("January 1, 2008")
    in_end_date = pd.to_datetime("December 31 2009")
    out_start_date = pd.to_datetime("January 1, 2010")
    out_end_date = pd.to_datetime("December 31 2011")

    # Trim to in sample range with a 30 day prior range for features
    max_lookback = 30
    start_date = in_start_date - datetime.timedelta(days=max_lookback)
    end_date = in_end_date

    # Normalize prices
    prices = get_data(symbols=[symbol], dates=pd.date_range(start_date, end_date), addSPY=False)
    prices = prices.fillna(method="ffill").fillna(method="bfill")
    prices["JPM"] = prices["JPM"] / prices["JPM"][0]

    plot_start = in_start_date + datetime.timedelta(days=180)
    plot_end = in_start_date + datetime.timedelta(days=250)

    """ Plot Bollinger Bands """

    moving_average_df = add_bollinger_band_indicator(prices, "JPM", True)

    # Plot
    plot_df = moving_average_df.loc[plot_start:plot_end,
              ["JPM", "moving_average", "bollinger_max", "bollinger_min"]].copy()

    plot_df.plot(title="Simple Moving Average\nBollinger Bands")


    def above_upper(i, df):
        return df.loc[i, "JPM"] > df.loc[i, "bollinger_max"]


    def below_lower(i, df):
        return df.loc[i, "JPM"] < df.loc[i, "bollinger_min"]


    for i in range(1, len(plot_df.index)):

        c_i = plot_df.index[i]
        l_i = plot_df.index[i - 1]

        if above_upper(l_i, plot_df) and not above_upper(c_i, plot_df):
            plt.axvline(x=c_i, color="black")
        if below_lower(l_i, plot_df) and not below_lower(c_i, plot_df):
            plt.axvline(x=c_i, color="lightblue")

    plt.ylabel("Normalized Price")
    plt.xlabel("Date")
    # plt.show()
    plt.savefig("bollinger_plot.png")

    """ MACD Plot """

    macd_df = add_macd(prices, symbol, True)
    plot_cols = ["JPM", "macd", "macd_signal_line"]
    line_cols = ["bullish_cross_over", "bearish_cross_over"]
    plot_df = macd_df.loc[plot_start:plot_end, plot_cols + line_cols]

    plot_df.loc[:, plot_cols].plot(secondary_y="JPM", title="Moving Average Convergence/Divergence")
    for i in plot_df.index:
        if plot_df.loc[i, "bullish_cross_over"]:
            plt.axvline(x=i, color="lightblue")
        if plot_df.loc[i, "bearish_cross_over"]:
            plt.axvline(x=i, color="black")
    plt.ylabel("Normalized Price")
    # plt.show()
    plt.savefig("macd_plot.png")

    """ Plot Momentum """

    momentum_df = add_momentum(prices, "JPM", True)
    plot_df = momentum_df.loc[plot_start:plot_end, :]

    plots = plot_df.plot(secondary_y="momentum", title="Momentum")

    plt.axhline(y=.1, color="black")

    plt.axhline(y=-.1, color="lightblue")
    plt.ylabel("Momentum Score")
    # plt.show()
    plt.savefig("momentum_plot.png")

    """ Stochastic D Plot """

    stoch_kd_df = add_stochastic_d(prices, symbol, add_helper_data=True)
    plot_df = stoch_kd_df.loc[plot_start:plot_end, ["JPM", "D"]]

    plot = plot_df.plot(secondary_y=["D"], title="Stochastic D")
    dir(plot)
    plt.axhline(y=.80, color="black")
    plt.axhline(y=.20, color="lightblue")
    plt.ylabel("Stochastic D Score")
    # plt.show()
    plt.savefig("stochastic_d_plot.png")
