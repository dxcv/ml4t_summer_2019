"""

Report

Please address each of these points/questions in your report, to be submitted as report.pdf

    In Experiment 1, estimate the probability of winning $80 within 1000 sequential bets. Explain your reasoning.
      * If the bet starts at 1 and doubles each loss, then the bet is represented as 2**l where l is the number of
        consecutive losses
      * Total losses are the sum of all prior bets, or sum(2**l_ for l_ in range(l))
      * And 2**l - sum(2**l_ for l_ in range(l)) == 1
      * Each win generates $1, no matter how many prior losses.
      * If the algorithm stops at $80, then one only needs to win 80 times out of 1000 trials.
      * The probability mass function calculates the probability of exactly k successes given n trials.
      * The probability of winning 80 out of 1000 sequential trials is the sum of the probability mass function
        for all of k in range(80, n+1). Where k is the exact number of wins and n is the total number of trials and
        range is exclusive.
      * This is a plot of the binomial distribution with the probability of 50% for winning per trial.
      * Based on this evidence I believe the probability of 80 or more wins approaches 100%.


    In Experiment 1, what is the estimated expected value of our winnings after 1000 sequential bets? Explain your
    reasoning. Go here to learn about expected value: https://en.wikipedia.org/wiki/Expected_value
      * Defined as "In probability theory, the expected value of a random variable, intuitively, is the long-run average
      value of repetitions of the same experiment it represents."
      *



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

"""
In Experiment 1, estimate the probability of winning $80 within 1000 sequential bets. Explain your reasoning.
"""

# 999 losses, 1 win
# 510 losses, 490 wins

# 2**(n-1)  # Where n is number of iterations since a win. n starts at 1.
# 2**n  # Where n is number of consecutive losses. n starts at 0

# 50 losses, what is the total of best lost?

# sum(2**n for n in range(50))


len(str(sum(2**n for n in range(999))))
len(str(2**103))

str(sum(2**n for n in range(999)))
str(2**999)

"{:,}".format(sum(2**n for n in range(500)))
"{:,}".format(2**500)


count = 0

count += 1

print "n", count
print "prior  ", sum(2**n for n in range(count))
print "current", 2**count

# No matter how many times you lose, every time you win, you get $1 added to the running total

# So to win 80, you only need to win 80 times

# What is the probability of winning 80 times out of 1,000 with a 50% probability of winning on each spin?

# .5**(1000-80)
# .5**(80./1000.)
# .5**(920./1000.)

# What is the sample space?
# https://math.stackexchange.com/questions/364986/probability-of-getting-exactly-2-heads-in-3-coins-tossed-with-order-not-importan
2**1000

# https://en.wikipedia.org/wiki/Binomial_distribution
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.binom.html

from scipy.special import binom
import numpy as np
from matplotlib import pyplot as plt


# The probability of getting exactly k successes in n trials is given by the probability mass function:

# Probability mass function


# Sum of entire probability mass function range where k >= 80 <= n
p = np.float128(0.5)  # prob of success
n = 1000  # # of trials
k = 1  # # of successes
print(np.sum(binom(n, k_) * p**k_ * (1-p)**(n-k_) for k_ in range(k, n+1)))

# Plot the probability distribution of at least k successes
n = 1000
result = []

for k in range(1, n+1):
    result.append(np.sum(binom(n, k_) * p ** k_ * (1 - p) ** (n - k_) for k_ in range(k, n + 1)))


plt.plot(result)
plt.axvline(x=8, color="green")
plt.title("Binomial Distribution w/ Unlimited Bankroll")
plt.ylabel("Probability")
plt.xlabel("Minimum Wins")
plt.text(9, .5, "At Least 80 Wins", verticalalignment='center')
plt.text(9, 1.02, "Probability approaches 100%", verticalalignment='center')
plt.savefig("martingale/experiment_1_binomial_distribution.png")
plt.close()

"""
In Experiment 1, what is the estimated expected value of our winnings after 1000 sequential bets? Explain your
reasoning. Go here to learn about expected value: https://en.wikipedia.org/wiki/Expected_value

  * Defined as "In probability theory, the expected value of a random variable, intuitively, is the long-run average 
  value of repetitions of the same experiment it represents."
  
  * The average final winnings is $80 after 1000 trials with 0 standard deviation. The expected value is therefore
    80 with ~100% confidence.

"""
from martingale.martingale import betting_scheme

np.random.seed(7646)
n_simulations = 1000
n_runs_per_sim = 1000

agg_winnings = np.zeros(shape=(n_simulations, n_runs_per_sim))

for i in range(n_simulations):
    agg_winnings[i, :] = betting_scheme(n_runs_per_sim)[:]

mean = np.mean(agg_winnings.T, axis=1)
std = np.std(agg_winnings.T, axis=1)

# Confidence interval of final value with 99% confidence(z = 2.576)
mean[-1]
std[-1]

"""
In Experiment 1,
does the standard deviation reach a maximum value then stabilize or converge as the number of sequential bets
increases? Explain why it does (or does not).
"""

np.argmax(std)
std[59]









