"""

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

"""
In Experiment 1, estimate the probability of winning $80 within 1000 sequential bets. Explain your reasoning.


What is the probability of winning once out of 1000 sequential bets?




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




