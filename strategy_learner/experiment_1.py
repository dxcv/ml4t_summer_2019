import StrategyLearner as sl
import pandas as pd
from util import get_data
from matplotlib import pyplot as plt
from marketsimcode import compute_portvals
import ManualStrategy as ms
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

    # Calculate benchmark
    prices = get_data(symbols=[symbol], dates=pd.date_range(in_start_date, in_end_date), addSPY=False)
    prices = prices.dropna()  # Instead of fill when creating trades
    benchmark_trades = pd.DataFrame(columns=[symbol], index=prices.index, data=0)
    benchmark_trades.iloc[0] = 1000

    # Train learner
    learner = sl.StrategyLearner(verbose=False, impact=0.0, dyna=200, epochs=3)  # constructor
    learner.addEvidence(symbol=symbol, sd=in_start_date, ed=in_end_date,
                        sv=100000, n=1)  # training phase

    # In sample policy test
    df_trades_leaner = learner.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date,
                                      sv=100000)  # testing phase

    # Get manual strategy
    df_trades_manual = ms.ManualStrategy.testPolicy(symbol="JPM", sd=in_start_date, ed=in_end_date, sv=100000,
                                                    return_signal=True)

    print "Benchmark Stats"
    benchmark_portvals = compute_portvals(symbol, benchmark_trades, commission=0, impact=0)
    benchmark_portvals.columns = ["Benchmark"]
    print calculate_statistics(benchmark_portvals)

    print "Strategy Learner"
    # Optimal portvals-+-
    in_portvals = compute_portvals(symbol, df_trades_leaner, commission=0, impact=0.0)
    in_portvals.columns = ["Strategy Learner"]
    print calculate_statistics(in_portvals)

    print "Manual Strategy"
    manual_portvals = compute_portvals(symbol, df_trades_manual, commission=0, impact=0.0)
    manual_portvals.columns = ["Manual Strategy"]
    print calculate_statistics(manual_portvals)

    # In sample plot
    plot_df = pd.concat([in_portvals, manual_portvals], axis=1)

    plot_df = plot_df / plot_df.iloc[0]
    plot_df.loc[:, ["Strategy Learner", "Manual Strategy"]].plot(color=["red", "green"], title="Strategy Learner vs Manual Strategy\nIn Sample")
    plt.ylabel("Normalized Price")
    plt.savefig("experiment_1_plot.png")

    # Manual strategy plot

"""
Benchmark Stats
{'avg_day_ret': Benchmark    0.000004
dtype: float64, 'cum_ret': Benchmark    0.00123
dtype: float64, 'std_day_ret': Benchmark    0.001613
dtype: float64}
Strategy Learner
{'avg_day_ret': Strategy Learner    0.000119
dtype: float64, 'cum_ret': Strategy Learner    0.06147
dtype: float64, 'std_day_ret': Strategy Learner    0.001056
dtype: float64}
Manual Strategy
{'avg_day_ret': Manual Strategy    0.000133
dtype: float64, 'cum_ret': Manual Strategy    0.06881
dtype: float64, 'std_day_ret': Manual Strategy    0.001549
dtype: float64}
"""
