import StrategyLearner as sl
import pandas as pd
from util import get_data
from matplotlib import pyplot as plt
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


def author():
    return 'cfarr31'


if __name__ == "__main__":
    symbol = "JPM"  # AAPL

    in_start_date = pd.to_datetime("January 1, 2008")
    in_end_date = pd.to_datetime("December 31, 2009")
    out_start_date = pd.to_datetime("January 1, 2010")
    out_end_date = pd.to_datetime("December 31, 2011")

    # Calculate benchmark
    in_prices = get_data(symbols=[symbol], dates=pd.date_range(in_start_date, in_end_date), addSPY=False)
    in_prices = in_prices.dropna()  # Instead of fill when creating trades
    in_benchmark_trades = pd.DataFrame(columns=[symbol], index=in_prices.index, data=0)
    in_benchmark_trades.iloc[0] = 1000

    out_prices = get_data(symbols=[symbol], dates=pd.date_range(out_start_date, out_end_date), addSPY=False)
    out_prices = out_prices.dropna()  # Instead of fill when creating trades
    out_benchmark_trades = pd.DataFrame(columns=[symbol], index=out_prices.index, data=0)
    out_benchmark_trades.iloc[0] = 1000

    # Train learner
    learner = sl.StrategyLearner(verbose=False, impact=0.95, dyna=200, epochs=3)  # constructor
    learner.addEvidence(symbol=symbol, sd=in_start_date, ed=in_end_date,
                        sv=100000, n=1)  # training phase

    print "State count"
    print len(learner._state_dict)

    # In sample policy test
    df_trades_in = learner.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date,
                                      sv=100000)  # testing phase

    # Out sample policy test
    df_trades_out = learner.testPolicy(symbol=symbol, sd=out_start_date, ed=out_end_date,
                                       sv=100000)  # testing phase

    print "Benchmark Stats"
    in_benchmark_portvals = compute_portvals(symbol, in_benchmark_trades, commission=0, impact=0)
    in_benchmark_portvals.columns = ["Benchmark"]
    print calculate_statistics(in_benchmark_portvals)

    print "In Sample Stats"
    # Optimal portvals-+-
    in_portvals = compute_portvals(symbol, df_trades_in, commission=0, impact=0)
    in_portvals.columns = ["Strategy Learner"]
    print calculate_statistics(in_portvals)

    print "Out of Sample Benchmark Stats"
    out_benchmark_portvals = compute_portvals(symbol, out_benchmark_trades, commission=0, impact=0)
    out_benchmark_portvals.columns = ["Benchmark"]
    print calculate_statistics(out_benchmark_portvals)

    print "Out Sample Stats"
    # Optimal portvals-+-
    out_portvals = compute_portvals(symbol, df_trades_out, commission=0, impact=0)
    out_portvals.columns = ["Strategy Learner"]
    print calculate_statistics(out_portvals)

    # In sample plot
    plot_df = pd.concat([in_portvals, in_benchmark_portvals], axis=1)

    plot_df = plot_df / plot_df.iloc[0]
    plot_df.loc[:, ["Strategy Learner", "Benchmark"]].plot(color=["red", "green"], title="Strategy Learner\nIn Sample")
    plt.ylabel("Normalized Price")
    for i in df_trades_in.index:
        if i not in plot_df.index:
            continue
        y = plot_df.loc[i, "Benchmark"]
        if df_trades_in.loc[i, symbol] == -1000:
            plt.vlines(x=i, ymin=y-.01, ymax=y, color="black")
        elif df_trades_in.loc[i, symbol] == 1000:
            plt.vlines(x=i, ymin=y, ymax=y+.01, color="blue")
    plt.savefig("in_sample_strategy_learner.png")

    # Out of sample plot
    plot_df = pd.concat([out_portvals, out_benchmark_portvals], axis=1)

    plot_df = plot_df / plot_df.iloc[0]
    plot_df.loc[:, ["Strategy Learner", "Benchmark"]].plot(color=["red", "green"], title="Strategy Learner\nOut of Sample")
    plt.ylabel("Normalized Price")
    for i in df_trades_out.index:
        if i not in plot_df.index:
            continue
        y = plot_df.loc[i, "Benchmark"]
        if df_trades_out.loc[i, symbol] == -1000:
            plt.vlines(x=i, ymin=y-.01, ymax=y, color="black")
        elif df_trades_out.loc[i, symbol] == 1000:
            plt.vlines(x=i, ymin=y, ymax=y+.01, color="blue")
    plt.savefig("out_sample_strategy_learner.png")

    # Create out of sample plot for comparison



"""
Benchmark
In Sample
{'avg_day_ret': Benchmark    0.000004
dtype: float64, 'cum_ret': Benchmark    0.00123
dtype: float64, 'std_day_ret': Benchmark    0.001613
dtype: float64}
out of sample
{'avg_day_ret': Benchmark   -0.000016
dtype: float64, 'cum_ret': Benchmark   -0.00834
dtype: float64, 'std_day_ret': Benchmark    0.000813
dtype: float64}

ManualStrategy
{'avg_day_ret': Manual    0.000099
dtype: float64, 'cum_ret': Manual    0.05029
dtype: float64, 'std_day_ret': Manual    0.001572
dtype: float64}
Out of Sample
{'avg_day_ret': Manual   -0.000017
dtype: float64, 'cum_ret': Manual   -0.00872
dtype: float64, 'std_day_ret': Manual    0.0008
dtype: float64}

StrategyLearner
In Sample Stats
{'avg_day_ret': In Sample    0.000126
dtype: float64, 'cum_ret': In Sample    0.06524
dtype: float64, 'std_day_ret': In Sample    0.001008
dtype: float64}
Out Sample Stats
{'avg_day_ret': Out Sample    0.000025
dtype: float64, 'cum_ret': Out Sample    0.01239
dtype: float64, 'std_day_ret': Out Sample    0.000574
dtype: float64}
"""


