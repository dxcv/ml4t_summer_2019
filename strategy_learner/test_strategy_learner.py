from StrategyLearner import StrategyLearner
from util import get_data

"""
For your report, trade only the symbol JPM. This will enable us to more easily compare results. We will test your 
learner with other symbols as well. You may use data from other symbols (such as SPY) to inform your strategy. 

The in sample/development period is January 1, 2008 to December 31 2009. The out of sample/testing period is January 1, 
2010 to December 31 2011. 
Starting cash is $100,000. 
Allowable positions are: 1000 shares long, 1000 shares short, 
0 shares. 
Benchmark: The performance of a portfolio starting with $100,000 cash, investing in 1000 shares of the 
symbol in use and holding that position. Include transaction costs. 
There is no limit on leverage. Transaction costs: 
Commission will always be $0.00, Impact may vary, and will be passed in as a parameter to the learner. 

"""

# symbol = "JPM"

import StrategyLearner as sl
import datetime as dt

import pandas as pd
from util import get_data
from matplotlib import pyplot as plt
import datetime
import os
import datetime as dt
from marketsimcode import compute_portvals

from random import seed
from numpy.random import seed as np_seed

seed(0)
np_seed(0)


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


if __name__ == "__main__":
    symbol = "JPM"  # AAPL

    in_start_date = pd.to_datetime("January 1, 2008")
    in_end_date = pd.to_datetime("December 31, 2009")
    out_start_date = pd.to_datetime("January 1, 2010")
    out_end_date = pd.to_datetime("December 31, 2011")

    # Calculate benchmark
    prices = get_data(symbols=[symbol], dates=pd.date_range(in_start_date, in_end_date), addSPY=False)
    prices = prices.dropna()  # Instead of fill when creating trades
    benchmark_trades = pd.DataFrame(columns=[symbol], index=prices.index, data=0)
    benchmark_trades.iloc[0] = 1000

    # Train learner
    learner = sl.StrategyLearner(verbose=False, impact=0.000, dyna=200, epochs=3)  # constructor
    learner.addEvidence(symbol=symbol, sd=in_start_date, ed=in_end_date,
                        sv=100000, n=1)  # training phase

    # In sample policy test
    df_trades_in = learner.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date,
                                      sv=100000)  # testing phase

    # Out sample policy test
    df_trades_out = learner.testPolicy(symbol=symbol, sd=out_start_date, ed=out_end_date,
                                       sv=100000)  # testing phase

    print "Benchmark Stats"
    benchmark_portvals = compute_portvals(symbol, benchmark_trades, commission=0, impact=0)
    benchmark_portvals.columns = ["Benchmark"]
    print calculate_statistics(benchmark_portvals)
    print "In Sample Stats"
    # Optimal portvals-+-
    in_portvals = compute_portvals(symbol, df_trades_in, commission=0, impact=0)
    in_portvals.columns = ["In Sample"]
    print calculate_statistics(in_portvals)
    print "Out Sample Stats"
    # Optimal portvals-+-
    out_portvals = compute_portvals(symbol, df_trades_out, commission=0, impact=0)
    out_portvals.columns = ["Out Sample"]
    print calculate_statistics(out_portvals)

    # Print Benchmark

    # Print in sample

    # Print out sample

    # plot_df = pd.concat([portvals, benchmark_portvals], axis=1)
    #
    # plot_df = plot_df / plot_df.iloc[0]
    # plot_df.loc[:, ["Optimal", "Benchmark"]].plot(color=["red", "green"], title="Theoretically Optimal Strategy")
    # plt.ylabel("Normalized Price")
    # plt.savefig("optimal_plot.png")


"""
Benchmark Stats
{'avg_day_ret': Benchmark    0.000004
dtype: float64, 'cum_ret': Benchmark    0.00123
dtype: float64, 'std_day_ret': Benchmark    0.001613
dtype: float64}
In Sample Stats
{'avg_day_ret': In Sample    0.00004
dtype: float64, 'cum_ret': In Sample    0.02024
dtype: float64, 'std_day_ret': In Sample    0.001079
dtype: float64}
Out Sample Stats
{'avg_day_ret': Out Sample    0.000032
dtype: float64, 'cum_ret': Out Sample    0.01635
dtype: float64, 'std_day_ret': Out Sample    0.000566
dtype: float64}
Bollinger only, n=2, pass 3/4 tests

n = 10
Benchmark Stats
{'avg_day_ret': Benchmark    0.000004
dtype: float64, 'cum_ret': Benchmark    0.00123
dtype: float64, 'std_day_ret': Benchmark    0.001613
dtype: float64}
In Sample Stats
{'avg_day_ret': In Sample    0.000047
dtype: float64, 'cum_ret': In Sample    0.02351
dtype: float64, 'std_day_ret': In Sample    0.001463
dtype: float64}
Out Sample Stats
{'avg_day_ret': Out Sample    0.000025
dtype: float64, 'cum_ret': Out Sample    0.0124
dtype: float64, 'std_day_ret': Out Sample    0.000727
dtype: float64}

Bollinger only, n=10, pass 3/4 tests

"""

"""
Divergence only

n=2
Benchmark Stats
{'avg_day_ret': Benchmark    0.000004
dtype: float64, 'cum_ret': Benchmark    0.00123
dtype: float64, 'std_day_ret': Benchmark    0.001613
dtype: float64}
In Sample Stats
{'avg_day_ret': In Sample    0.00002
dtype: float64, 'cum_ret': In Sample    0.0099
dtype: float64, 'std_day_ret': In Sample    0.0009
dtype: float64}
Out Sample Stats
{'avg_day_ret': Out Sample    0.00002
dtype: float64, 'cum_ret': Out Sample    0.01009
dtype: float64, 'std_day_ret': Out Sample    0.000486
dtype: float64}

Divergence only, n=2 passed 2/4 tests

n=10
Benchmark Stats
{'avg_day_ret': Benchmark    0.000004
dtype: float64, 'cum_ret': Benchmark    0.00123
dtype: float64, 'std_day_ret': Benchmark    0.001613
dtype: float64}
In Sample Stats
{'avg_day_ret': In Sample    0.000006
dtype: float64, 'cum_ret': In Sample    0.00247
dtype: float64, 'std_day_ret': In Sample    0.001243
dtype: float64}
Out Sample Stats
{'avg_day_ret': Out Sample   -0.000008
dtype: float64, 'cum_ret': Out Sample   -0.00422
dtype: float64, 'std_day_ret': Out Sample    0.00063
dtype: float64}

Divergence only, n=10 passed 2/4 tests

"""
