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
      * The average final winnings is $80 after 1000 trials with 0 standard deviation. The expected value is therefore
      80 with ~100% confidence.

    In Experiment 1,
    does the standard deviation reach a maximum value then stabilize or converge as the number of sequential bets
    increases? Explain why it does (or does not).
      * The standard deviation appears to hit a maximum value and then drops as the probability of hitting 80 total
       wins increases. It finally reaches zero as the probability of hitting 80 total wins approaches 100%
      * In this plot it shows a forward looking rolling average of the standard deviation, for smoothing. After
      1000 simulations the high variability shows a very sporadic standard deviation, by smoothing it shows the trend
      of the std more clearly.
      * The green line shows where the probability of hitting 80 total wins reaches 99.5%, in this case around after 194
      trials. The std clearly stabilizes
       prior to this within the 1000 runs, but if more runs were tested then it could theoretically stabilize at
       various trial counts.


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
# Total sample space: 2**1000

# https://en.wikipedia.org/wiki/Binomial_distribution
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.binom.html

from scipy.special import binom
import numpy as np
from matplotlib import pyplot as plt


# The probability of getting exactly k successes in n trials is given by the probability mass function:

# Probability mass function

# Sum of entire probability mass function range where k >= 80 <= n
p = np.float128(0.5)  # prob of success
n = 200  # # of trials
k = 80  # # of successes
print(np.sum(binom(n, k_) * p**k_ * (1-p)**(n-k_) for k_ in range(k, n+1)))

# Plot the probability distribution of at least k successes
p = np.float128(0.5)  # prob of success
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
import numpy as np

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
std[np.argmax(std)]

# Std is largely impacted by the max dollar threshold of $80
# Once it's reached, the value stabilizes.


# Std correlates with the probability of reaching 80 rolls

# Reproduce figure 2 data and plot the rolling mean of the std (to smooth over randomness)
import pandas as pd
from matplotlib import pyplot as plt


p = np.float128(0.5)  # prob of success
n = 1000
result = []
k = 80


# Chance of hitting 80 wins
for n_ in range(1, n+1):
    result.append(np.sum(binom(n_, k_) * p**k_ * (1-p)**(n_-k_) for k_ in range(k, n_+1)))


std_df = pd.DataFrame(std)


rolling_std_df = std_df.iloc[::-1].rolling(50).mean().iloc[::-1]
plt.plot(rolling_std_df)
plt.plot(np.array(result) * int(np.max(rolling_std_df)))
plt.axvline(x=np.min(np.argwhere(np.array(result) > .99)), color="green")
plt.xlim(left=0, right=300)
plt.title("Winnings Std vs >=80 Wins Probability")
plt.ylabel("Forward Rolling Std (blue) \n >=80 Wins Prob (orange)")
plt.xlabel("Trials Count")
plt.show()

plt.close()

"""
    In Experiment 2, estimate the probability of winning $80 within
    1000 sequential bets. Explain your reasoning using the experiment. (not based on plots)
    
    * Same question, put differently, "What is the probability that the simulation will stay above 
      $256 before getting 80 wins?"
    
    * The probable winning amount at any point is $0, with even probability of a win or loss.
    * Bet size can be calculated by 2**l where l is number of consecutive losses. Total loss is
      2**l - 1 calculated prior to the bet amount for that trial.
    * Based on the betting scheme, the maximum number of consecutive losses before hitting bankroll
      is log base 2 of the bank roll.
    * If at any time there are 8 consecutive losses then the gambler stops. The probability of hitting
      8 consecutive losses is .5**8
      
    * Based on the experiments, 74%
    
"""

# What is the probability that the simulation will stay above $256 before getting 80 wins?
# How many sequential losses, given total prior wins, lead to hitting -256?
# What is the highest number of consecutive losses allowed before hitting bank roll at any trial count? Assuming
# evenly hitting losses and wins up to that point.
from math import floor
from martingale.martingale import betting_scheme
p = 0.5
bank_roll = 256

n_trials = 10
probable_total_winnings = n_trials * p
# Allowable consecutive losses before likely hitting bankroll
int(floor(np.log2(bank_roll + probable_total_winnings)))
# Probability of hitting the allowed losses at each step
# Step 1
# Prob win .5
# Prob loss .5
# Step 2
# Prob win 1
# Prob loss 1
# Step 3
# 1.5 1.5
# Step 4
# 2 2
# Probable total winnings are always 0
# Allowed losses for bankroll 8, then max is reached on 9 so the bet can't be made.... so call it 9
2**8 - 1
int(floor(np.log2(bank_roll)))

# Must use cumulative bets
sum(list(2**i for i in range(8)))
2**8



# Probability of 9 consecutive losses is...
.5**8
# How many chances are there to get 9 con losses
# using 99% probability as ceiling (191 trials)
.5**8 * (191 - 9)
# 35.5% probability of losing bankroll


# .19% probability

# This seems low... how many times does it happen?
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
plt.show()

# Anytime the winnings total drops, it was a loss

agg_winnings_df = pd.DataFrame(agg_winnings.T)

trial_loss_history = agg_winnings_df == 80
trial_loss_history = trial_loss_history.rolling(2).apply(lambda x: x[0] < x[1])

# Rolling sum of hitting bankroll (this can help to estimate probability)
trial_loss_history.iloc[0, :] = 0.0
trial_loss_history = trial_loss_history.cumsum()
trial_loss_history.sum(axis=1).plot()

plt.show()


trial_loss_history

# Take the final row, how many reached 80 in winnings?  / total == prob
trial_loss_history.iloc[-1, :].sum() / trial_loss_history.shape[1]
# 74% of them
# Experiment shows a 26% probability of loss
# If this experiment was ran over an over I would expect the prob to be closer to 36%.

trial_loss_history.rolling(5).sum().plot()
plt.show()
# At trial 1, 50/50 chance of getting 1, 0% chance of losing




"""

    In Experiment 2,
    what is the estimated expected value of our winnings after 1000 sequential bets? Explain your reasoning. (not
    based on plots)
"""
"""

    In Experiment 2, does the standard deviation reach a maximum value then stabilize or converge as
    the number of sequential bets increases? Explain why it does (or does not). Include figures 1 through 5.


"""