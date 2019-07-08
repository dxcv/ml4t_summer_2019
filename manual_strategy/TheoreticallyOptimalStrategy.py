"""Code implementing a TheoreticallyOptimalStrategy object (details below). It should implement testPolicy() which
returns a trades data frame (see below). The main part of this code should call marketsimcode as necessary to
generate the plots used in the report.

Please add in an author function to each file."""

from util import get_data, plot_data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import os
import datetime as dt
from marketsimcode import compute_portvals

# os.chdir(os.path.join(os.getcwd(), "manual_strategy"))


def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    prices = get_data(symbols=[symbol], dates=pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna()  # Instead of fill when creating trades

    # Optimal strategy, buy if next day goes up and not in the bought position
    #  Sell if next day goes down and not in the sold position
    #  Hold if next day is flat

    prices["signal"] = prices.rolling(window=2).apply(lambda x: -1 if x[-1] < x[0] else 1 if x[0] < x[1] else 0).shift(
        -1).fillna(0).astype(int)

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


def author():
    return 'cfarr31'  # replace tb34 with your Georgia Tech username.


sym = ["JPM"]

in_start_date = pd.to_datetime("January 1, 2008")
in_end_date = pd.to_datetime("December 31 2009")
out_start_date = pd.to_datetime("January 1, 2010")
out_end_date = pd.to_datetime("December 31 2011")

# Trim to in sample range with a 30 day prior range for features
# max_lookback = 0
# start_date = in_start_date - datetime.timedelta(days=max_lookback)
# end_date = in_end_date


# Benchmark trades
# Benchmark: The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of JPM and holding
# that position.
prices = get_data(symbols=["JPM"], dates=pd.date_range(in_start_date, in_end_date), addSPY=False)
prices = prices.dropna()  # Instead of fill when creating trades
benchmark_trades = pd.DataFrame(columns=["JPM"], index=prices.index, data=0)
benchmark_trades.iloc[0] = 1000

# Theoretically optimal strategy
trades = testPolicy("JPM", sd=in_start_date, ed=in_end_date)


# Benchmark portvals
benchmark_portvals = compute_portvals(benchmark_trades, commission=0, impact=0)
benchmark_portvals.columns = ["Benchmark"]

# Optimal portvals
portvals = compute_portvals(trades, commission=0, impact=0)
portvals.columns = ["Optimal"]

plot_df = pd.concat([portvals, benchmark_portvals], axis=1)

plot_df = plot_df / plot_df.iloc[0]
plot_df.loc[:, ["Optimal", "Benchmark"]].plot(color=["red", "green"], title="Theoretically Optimal Strategy")
plt.show()


"""
TODO's
You should also report in text:

    Cumulative return of the benchmark and portfolio
    Stdev of daily returns of benchmark and portfolio
    Mean of daily returns of benchmark and portfolio
"""

