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


Betting scheme
    episode_winnings = $0
    while episode_winnings < $80:
        won = False
        bet_amount = $1
        while not won
            wager bet_amount on black
            won = result of roulette wheel spin
            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2


Build simulator

Revise the code in martingale.py to simulate 1000 successive bets on spins of the roulette wheel using the betting
scheme outlined above. You should test for the results of the betting events by making successive calls to the
get_spin_result(win_prob) function. Note that you'll have to update the win_prob parameter according to the correct
probability of winning. You can figure that out by thinking about how roulette works (see wikipedia link above).

Track your winnings by storing them in a numpy array. You might call that array winnings where winnings[0] should be
set to 0 (just before the first spin). winnings[1] should reflect the total winnings after the first spin and so on.
For a particular episode if you ever hit $80 in winnings, stop betting and just fill the data forward with the value
80.


Experiment 1: Explore the strategy and make some charts

Now we want you to run some experiments to determine how well the betting strategy works. The approach we're going to
take is called Monte Carlo simulation where the idea is to run a simulator over and over again with randomized inputs
and to assess the results in aggregate. Skip to the "report" section below to which specific properties of the
strategy we want you to evaluate.

For the following charts, and for all charts in this class you should use python's matplotlib library. Your submitted
project should include all of the code necessary to generate the charts listed in your report. You should configure
your code to write the figures to .png files. Do not allow your code to create a window that displays images. If it
does you will receive a penalty.

    Figure 1: Run your simple simulator 10 times and track the winnings, starting from 0 each time. Plot all 10 runs
    on one chart using matplotlib functions. The horizontal (X) axis should range from 0 to 300, the vertical (Y)
    axis should range from -256 to +100. Note that we will not be surprised if some of the plot lines are not visible
    because they exceed the vertical or horizontal scales.
    Figure 2: Run your simple simulator 1000 times. Plot the
    mean value of winnings for each spin using the same axis bounds as Figure 1. Add an additional line above and
    below the mean at mean+standard deviation, and mean-standard deviation of the winnings at each point.
    Figure 3:
    Use the same data you used for Figure 2, but plot the median instead of the mean. Be sure to include the standard
    deviation lines above and below the median as well.

For all of the above charts and experiments, if and when the target $80 winnings is reached, stop betting and allow
the $80 value to persist from spin to spin.

Experiment 2: A more realistic gambling simulator

You may have noticed that the strategy actually works pretty well, maybe better than you expected. One reason for
this is that we were allowing the gambler to use an unlimited bank roll. In this experiment we're going to make
things more realistic by giving the gambler a $256 bank roll. If he or she runs out of money, bzzt, that's it. Repeat
the experiments above with this new condition. Note that once the player has lost all of their money (i.e.,
episode_winnings reaches -256) stop betting and fill that number (-256) forward. An important corner case to be sure
you handle is the situation where the next bet should be $N, but you only have $M (where M<N). Make sure you only bet
$M. Here are the two charts to create:

    Figure 4: Run your realistic simulator 1000 times. Plot the mean value of winnings for each spin using the same
    axis bounds as Figure 1. Add an additional line above and below the mean at mean+standard deviation,
    and mean-standard deviation of the winnings at each point.
    Figure 5: Use the same data you used for Figure 4,
    but use the median instead of the mean. Be sure to include the standard deviation lines above and below the
    median as well.


Report

Please address each of these points/questions in your report, to be submitted as report.pdf

    In Experiment 1, estimate the probability of winning $80 within 1000 sequential bets. Explain your reasoning.
      https://math.stackexchange.com/questions/1287196/probability-of-tossing-a-coin-1000-times-and-coming-up-with-heads
    In Experiment 1, what is the estimated expected value of our winnings after 1000 sequential bets? Explain your
    reasoning. Go here to learn about expected value: https://en.wikipedia.org/wiki/Expected_value
      50% probability, $1 per win. Does it stop at $80? If not, then $500.
    In Experiment 1,
    does the standard deviation reach a maximum value then stabilize or converge as the number of sequential bets
    increases? Explain why it does (or does not).




    In Experiment 2, estimate the probability of winning $80 within
    1000 sequential bets. Explain your reasoning using the experiment. (not based on plots)
    In Experiment 2,
    what is the estimated expected value of our winnings after 1000 sequential bets? Explain your reasoning. (not
    based on plots)
    In Experiment 2, does the standard deviation reach a maximum value then stabilize or converge as
    the number of sequential bets increases? Explain why it does (or does not). Include figures 1 through 5.



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
    plt.xlim(left=0, right=300)
    plt.ylim(bottom=-256, top=100)
    plt.title("Figure 1")
    plt.xlabel("Number of Spins")
    plt.ylabel("Total Winnings")
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
    plt.xlim(left=0, right=300)
    plt.ylim(bottom=-256, top=100)
    plt.title("Figure 2")
    plt.xlabel("Number of Spins")
    plt.ylabel("Mean Winnings (+- std)")
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
    plt.xlim(left=0, right=300)
    plt.ylim(bottom=-256, top=100)
    plt.title("Figure 3")
    plt.xlabel("Number of Spins")
    plt.ylabel("Median Winnings (+- std)")
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
    plt.xlim(left=0, right=300)
    plt.ylim(bottom=-256, top=100)
    plt.title("Figure 4")
    plt.xlabel("Number of Spins")
    plt.ylabel("Mean Winnings (+- std)")
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
    plt.xlim(left=0, right=300)
    plt.ylim(bottom=-256, top=100)
    plt.title("Figure 5")
    plt.xlabel("Number of Spins")
    plt.ylabel("Median Winnings (+- std)")
    plt.savefig("martingale/figure_5.png")
    plt.close()

    return


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
