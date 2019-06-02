"""Assess a betting strategy. 			  		 			 	 	 		 		 	  		   	  			  	

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

import numpy as np
from matplotlib import pyplot as plt


def author():
    return 'cfarr31'  # replace tb34 with your Georgia Tech username.


def gtid():
    return 90347082  # replace with your GT ID number


def get_spin_result(win_prob):
    return np.random.random() <= win_prob


def test_code():
    win_prob = 0.60  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    print get_spin_result(win_prob)  # test the roulette spin


# add your code here to implement the experiments

def betting_scheme(n, max_winnings=80, bank_roll=None):

    WIN_PROB = 0.50
    winnings = np.zeros(shape=(n,))
    episode_winnings = 0
    bet_amount = 1

    for i in range(n):

        won = get_spin_result(WIN_PROB)

        if won:
            episode_winnings = episode_winnings + bet_amount
            bet_amount = 1
        else:
            episode_winnings = episode_winnings - bet_amount
            bet_amount = bet_amount * 2

        winnings[i] = episode_winnings

        if bank_roll is not None and (episode_winnings - bet_amount) < (bank_roll * -1):
            winnings[i:] = episode_winnings
            break

        if episode_winnings >= max_winnings:
            winnings[i:] = episode_winnings
            break

    return winnings


def figure_1():
    """
    Figure 1: Run your simple simulator 10 times and track the winnings, starting from 0 each time. Plot all 10 runs
    on one chart using matplotlib functions. The horizontal (X) axis should range from 0 to 300, the vertical (Y)
    axis should range from -256 to +100. Note that we will not be surprised if some of the plot lines are not visible
    because they exceed the vertical or horizontal scales.
    """

    np.random.seed(7646)
    n_simulations = 10
    n_runs_per_sim = 1000
    agg_winnings = np.zeros(shape=(n_simulations, n_runs_per_sim))

    for i in range(n_simulations):
        agg_winnings[i, :] = betting_scheme(n_runs_per_sim)[:]

    plt.plot(agg_winnings.T, color="gray")
    plt.legend(["Total Winnings"])
    plt.xlim(left=0, right=300)
    plt.ylim(bottom=-256, top=100)
    plt.title("Figure 1")
    plt.xlabel("Number of Spins")
    plt.ylabel("Dollars")
    plt.savefig("martingale/figure_1.png")
    plt.close()

    return


def figure_2():
    """
    Figure 2: Run your simple simulator 1000 times. Plot the
    mean value of winnings for each spin using the same axis bounds as Figure 1. Add an additional line above and
    below the mean at mean+standard deviation, and mean-standard deviation of the winnings at each point.
    """

    np.random.seed(7646)
    n_simulations = 1000
    n_runs_per_sim = 1000

    agg_winnings = np.zeros(shape=(n_simulations, n_runs_per_sim))

    for i in range(n_simulations):
        agg_winnings[i, :] = betting_scheme(n_runs_per_sim)[:]

    mean = np.mean(agg_winnings.T, axis=1)
    std = np.std(agg_winnings.T, axis=1)
    plt.plot(mean, color="gray")
    plt.plot(mean + std, color="green")
    plt.plot(mean - std, color="red")
    plt.legend(["Mean Winnings", "Mean + std", "Mean - std"])
    plt.xlim(left=0, right=300)
    plt.ylim(bottom=-256, top=100)
    plt.title("Figure 2")
    plt.xlabel("Number of Spins")
    plt.ylabel("Dollars")
    plt.savefig("martingale/figure_2.png")
    plt.close()

    return


def figure_3():
    """
    Figure 3:
    Use the same data you used for Figure 2, but plot the median instead of the mean. Be sure to include the standard
    deviation lines above and below the median as well.
    """

    np.random.seed(7646)
    n_simulations = 1000
    n_runs_per_sim = 1000

    agg_winnings = np.zeros(shape=(n_simulations, n_runs_per_sim))

    for i in range(n_simulations):
        agg_winnings[i, :] = betting_scheme(n_runs_per_sim)[:]

    median = np.median(agg_winnings.T, axis=1)
    std = np.std(agg_winnings.T, axis=1)
    plt.plot(median, color="gray")
    plt.plot(median + std, color="green")
    plt.plot(median - std, color="red")
    plt.legend(["Median Winnings", "Median + std", "Median - std"])
    plt.xlim(left=0, right=300)
    plt.ylim(bottom=-256, top=100)
    plt.title("Figure 3")
    plt.xlabel("Number of Spins")
    plt.ylabel("Dollars")
    plt.savefig("martingale/figure_3.png")
    plt.close()

    return


def figure_4():
    """
    Figure 4: Run your realistic simulator 1000 times. Plot the mean value of winnings for each spin using the same
    axis bounds as Figure 1. Add an additional line above and below the mean at mean+standard deviation,
    and mean-standard deviation of the winnings at each point.
    """

    np.random.seed(7646)
    n_simulations = 1000
    n_runs_per_sim = 1000
    bank_roll = 256

    agg_winnings = np.zeros(shape=(n_simulations, n_runs_per_sim))

    for i in range(n_simulations):
        agg_winnings[i, :] = betting_scheme(n_runs_per_sim, bank_roll=bank_roll)[:]

    mean = np.mean(agg_winnings.T, axis=1)
    std = np.std(agg_winnings.T, axis=1)
    plt.plot(mean, color="gray")
    plt.plot(mean + std, color="green")
    plt.plot(mean - std, color="red")
    plt.legend(["Mean Winnings", "Mean + std", "Mean - std"])
    plt.xlim(left=0, right=300)
    plt.ylim(bottom=-256, top=100)
    plt.title("Figure 4")
    plt.xlabel("Number of Spins")
    plt.ylabel("Dollars")
    plt.savefig("martingale/figure_4.png")
    plt.close()

    return


def figure_5():
    """
    Figure 5: Use the same data you used for Figure 4,
    but use the median instead of the mean. Be sure to include the standard deviation lines above and below the
    median as well.
    """

    np.random.seed(7646)
    n_simulations = 1000
    n_runs_per_sim = 1000
    bank_roll = 256

    agg_winnings = np.zeros(shape=(n_simulations, n_runs_per_sim))

    for i in range(n_simulations):
        agg_winnings[i, :] = betting_scheme(n_runs_per_sim, bank_roll=bank_roll)[:]

    median = np.median(agg_winnings.T, axis=1)
    std = np.std(agg_winnings.T, axis=1)
    plt.plot(median, color="gray")
    plt.plot(median + std, color="green")
    plt.plot(median - std, color="red")
    plt.legend(["Median Winnings", "Median + std", "Median - std"])
    plt.xlim(left=0, right=300)
    plt.ylim(bottom=-256, top=100)
    plt.title("Figure 5")
    plt.xlabel("Number of Spins")
    plt.ylabel("Dollars")
    plt.savefig("martingale/figure_5.png")
    plt.close()

    return


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
