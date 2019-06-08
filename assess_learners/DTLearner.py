""" API
import DTLearner as dt
learner = dt.DTLearner(leaf_size = 1, verbose = False)  # constructor
learner.addEvidence(Xtrain, Ytrain)  # training step
Y = learner.query(Xtest)  # query
"""

# TODO for analysis
# Use 5-fold CV (implement KFold)
# Load instanbul data
#  The overall objective is to predict what the return for the MSCI Emerging Markets (EM) index will be on the basis
#  of the other index returns.
# Create X and y data
# Create arrays of indices for sub-setting data into splits

# Load istanbul data
import numpy as np


class DTLearner:
    NODE_INDEX = 0
    FEAT_INDEX = 1
    SPLIT_VALUE_INDEX = 2
    LEFT_INDEX = 3
    RIGHT_INDEX = 4

    def __init__(self, leaf_size=1, verbose=False):
        self.tree_data = None
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'cfarr31'  # replace tb34 with your Georgia Tech username.

    def addEvidence(self, Xtrain, Ytrain):
        self.tree_data = self.build_tree(Xtrain, Ytrain, leaf_size=self.leaf_size, verbose=self.verbose)
        return self

    @staticmethod
    def build_tree(x, y, depth=0, leaf_size=1, verbose=False):
        if x.shape[0] <= leaf_size:
            return np.array([[depth, np.nan, np.mean(y), np.nan, np.nan]])
        if np.std(y) == 0:
            return np.array([[depth, np.nan, np.mean(y), np.nan, np.nan]])

        feature_scores = np.array([np.abs(np.float(np.correlate(x[:, j], y))) for j in range(x.shape[1] - 1)])

        feature_sort_i = np.argsort(feature_scores)[::-1]

        # Loop through feature_sort_i until find one with std > 0
        # If none are available, return leaf
        feature = np.nan
        for i in feature_sort_i:
            if np.std(x[:, i]) == 0:
                # If zero std in feature split, continue
                continue
            elif np.all(x[:, i] < np.median(x[:, i])) or not np.any(x[:, i] < np.median(x[:, i])):
                # If none or all are less than median, continue
                continue
            else:
                feature = i
                break
        if np.isnan(feature):
            return np.array([[depth, np.nan, np.mean(y), np.nan, np.nan]])

        split_value = np.median(x[:, feature])

        left_index = x[:, feature] < split_value
        right_index = x[:, feature] >= split_value

        left_tree = DTLearner.build_tree(x[left_index, :], y[left_index], depth + 1, leaf_size, verbose)
        right_tree = DTLearner.build_tree(x[right_index, :], y[right_index],
                                          depth + left_tree.shape[0] + 1, leaf_size, verbose)

        root = np.array([[depth, feature, split_value, depth + 1, depth + left_tree.shape[0] + 1]])

        return np.vstack([root, left_tree, right_tree])

    def query(self, Xtrain):
        if self.tree_data is None:
            raise AssertionError("Attempting to query an untrained model")

        predictions = []

        for i in range(len(Xtrain)):
            node = 0

            while True:
                feature = np.int(self.tree_data[node, DTLearner.FEAT_INDEX]) if not np.isnan(
                    self.tree_data[node, DTLearner.FEAT_INDEX]) else np.nan
                split_value = self.tree_data[node, DTLearner.SPLIT_VALUE_INDEX]
                if np.isnan(feature):
                    predictions.append(split_value)
                    break
                if Xtrain[i, feature] < split_value:
                    node = np.int(self.tree_data[node, DTLearner.LEFT_INDEX])
                else:
                    node = np.int(self.tree_data[node, DTLearner.RIGHT_INDEX])
        return np.array(predictions)

# import pandas as pd
#
# df = pd.read_csv("assess_learners/Data/Istanbul.csv")
#
# train_len = int(len(df) * .8)
#
# train, test = df.iloc[:train_len, 1:], df.iloc[train_len:, 1:]
#
# model = DTLearner(leaf_size=4)
#
# model.addEvidence(train.iloc[:, :-1].values, train.iloc[:, -1].values)
#
# model.tree_data.shape
#
# y_test = test.iloc[:, -1].values
#
# predictions = model.query(test.iloc[:, :-1].values)
#
# # TODO RTLearner: RT video 2, 46 minutes in
#
# print np.sqrt(np.mean((y_test - predictions)**2))
#
# # TODO Start here!
