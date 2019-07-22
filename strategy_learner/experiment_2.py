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

    # Train learner
    learner = sl.StrategyLearner(verbose=False, impact=0.0, dyna=200, epochs=3)  # constructor
    learner.addEvidence(symbol=symbol, sd=in_start_date, ed=in_end_date,
                        sv=100000, n=1)  # training phase

    # In sample policy test
    df_trades = learner.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date,
                                   sv=100000)  # testing phase

    # Train impact learner
    impact_learner = sl.StrategyLearner(verbose=False, impact=0.005, dyna=200, epochs=3)  # constructor
    impact_learner.addEvidence(symbol=symbol, sd=in_start_date, ed=in_end_date,
                               sv=100000, n=1)  # training phase

    # In sample policy test
    df_trades_impact = impact_learner.testPolicy(symbol=symbol, sd=in_start_date, ed=in_end_date,
                                                 sv=100000)  # testing phase

    print "No Impact Stats"
    # Optimal portvals-+-
    portvals = compute_portvals(symbol, df_trades, commission=0, impact=0)
    portvals.columns = ["Standard Learner"]
    print calculate_statistics(portvals)
    print "Total Trades: %s" % df_trades.apply(lambda x: x != 0)["JPM"].sum()

    print "Impact Stats"
    # Optimal portvals-+-
    impact_portvals = compute_portvals(symbol, df_trades_impact, commission=0, impact=0)
    impact_portvals.columns = ["Impact Learner"]
    print calculate_statistics(impact_portvals)
    print "Total Trades: %s" % df_trades_impact.apply(lambda x: x != 0)["JPM"].sum()

    # In sample plot
    plot_df = pd.concat([portvals, impact_portvals], axis=1)

    plot_df = plot_df / plot_df.iloc[0]
    plot_df.loc[:, ["Standard Learner", "Impact Learner"]].plot(color=["red", "green"], title="Strategy Learner\nImpact Analysis")
    plt.ylabel("Normalized Price")
    plt.savefig("experiment_2_plot.png")


"""
No Impact Stats
{'avg_day_ret': Standard Learner    0.000119
dtype: float64, 'cum_ret': Standard Learner    0.06147
dtype: float64, 'std_day_ret': Standard Learner    0.001056
dtype: float64}
Total Trades: 201
Impact Stats
{'avg_day_ret': Impact Learner    0.000146
dtype: float64, 'cum_ret': Impact Learner    0.07615
dtype: float64, 'std_day_ret': Impact Learner    0.001153
dtype: float64}
Total Trades: 143
"""