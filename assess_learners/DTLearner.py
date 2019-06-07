""" API
import DTLearner as dt
learner = dt.DTLearner(leaf_size = 1, verbose = False)  # constructor
learner.addEvidence(Xtrain, Ytrain)  # training step
Y = learner.query(Xtest)  # query
"""
import numpy as np

"""

def build_tree(data):
    leaf = "factor"
    if data.shape[0] == 1:
        return [leaf, data.y, np.nan, np.nan]
    if all(data.ysame):
        return [leaf, data.y, np.nan, np.nan]
    else:
        i = "what"
        # Append what to what?
        # determine best feature ito split on
        SplitVal = data[:, i].median()
        lefttree = build_tree(data[data[:, i] <= SplitVal])
        righttree = build_tree(data[data[:, i] > SplitVal])
        root = [i, SplitVal, 1, lefttree.shape[0] + 1]
        return (append(root, lefttree, righttree))
"""

"""
Step through the algorithm...

Root = 0
Find best factor to split
Calculate split value
Divide data into left and right of split

Go to left split
Increment leaf by 1
Find best factor to split
Calculate split value
Divide data into left and right of split

Leaf is reached in data
Append nan, y data, nan, nan to tree data



"""



# TODO
# Use 5-fold CV (implement KFold)
# Load instanbul data
#  The overall objective is to predict what the return for the MSCI Emerging Markets (EM) index will be on the basis
#  of the other index returns.
# Create X and y data
# Create arrays of indices for sub-setting data into splits

# Load istanbul data

import pandas as pd
import numpy as np

df = pd.read_csv("assess_learners/Data/Istanbul.csv")


def build_tree(x, y, depth):
    if x.shape[0] == 1:
        return np.array([[depth, np.nan, np.mean(y), np.nan, np.nan]])
    if np.std(y) == 0:
        return np.array([[depth, np.nan, np.mean(y), np.nan, np.nan]])

    feature = np.argmax([np.float(np.correlate(x[:, j], y)) for j in range(x.shape[1] - 1)])

    split_value = np.median(x[:, feature])
    left_tree = build_tree(x[x[:, feature] <= split_value, :], y[x[:, feature] <= split_value], depth + 1)
    right_tree = build_tree(x[x[:, feature] > split_value, :], y[x[:, feature] > split_value], depth + left_tree.shape[0] + 1)

    root = np.array([[depth, feature, split_value, depth + 1, depth + left_tree.shape[0] + 1]])

    return np.vstack([root, left_tree, right_tree])


x_ = df.values[:10, 1:-1]
y_ = df.values[:10, -1]
my_tree = build_tree(x_, y_, 0)


# Query the tree

# Start with node 0
# If feature <= split value, go to left tree
# else, go to right tree

# Until feature is nan

NODE_INDEX = 0
FEAT_INDEX = 1
SPLIT_VALUE_INDEX = 2
LEFT_INDEX = 3
RIGHT_INDEX = 4

i = 0

i += 1

example_x = df.iloc[i, 1:-1].values
example_y = df.iloc[i, -1]


node = 0

while True:
    feature = np.int(my_tree[node, FEAT_INDEX]) if not np.isnan(my_tree[node, FEAT_INDEX]) else np.nan
    split_value = my_tree[node, SPLIT_VALUE_INDEX]

    if np.isnan(feature):
        final_answer = split_value
        break
    if example_x[feature] <= split_value:
        node = np.int(my_tree[node, LEFT_INDEX])
    else:
        node = np.int(my_tree[node, RIGHT_INDEX])

print("Example:", i)
print("Prediction:", final_answer)
print("Actual:", example_y)







my_tree.shape

"""
How to determine “best” feature?

Goal: Divide and conquer

Group data into most similar groups.


Approaches:
•Information gain: Entropy
•Information gain: Correlation
•Information gain: GiniIndex
https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

"""



