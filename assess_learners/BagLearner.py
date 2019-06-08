


class BagLearner:
    def __init__(self, learner, kwargs, bags, boost, verbose):
        pass

    def author(self):
        return 'cfarr31'  # replace tb34 with your Georgia Tech username.

    def addEvidence(self, Xtrain, Ytrain):
        pass

    def query(self, Xtest):
        pass

# Bagging: choose 60% of train examples randomly, resample until matching original example n
# Boosting: after each bag is created, test on all of the training data. when selecting the 60%
#   to sample from, weight based on error for choosing

import assess_learners.DTLearner as dt
import assess_learners.RTLearner as rt
import pandas as pd
import numpy as np
from time import time

df = pd.read_csv("assess_learners/Data/Istanbul.csv")


data = df.iloc[:, 1:].values


# Split all indices into array chunks
# Implement cross-validation


def get_cv_splits(data, k=10, random_state=None):
    all_indices = np.arange(0, data.shape[0])
    if random_state is not None:
        np.random.seed(random_state)
    else:
        np.random.seed(np.int(time()))
    np.random.shuffle(all_indices)
    test_splits = np.array_split(all_indices, k)

    # Create tuple of train, test indices
    validation_splits = []
    for i in range(len(test_splits)):
        test_i = test_splits[i]
        train_i = np.concatenate([test_splits[i_] for i_ in range(len(test_splits)) if i_ != i])
        validation_splits.append((train_i, test_i))
    return validation_splits


cv_scores = []

for train, test in get_cv_splits(data, random_state=None):
    x_train, y_train = data[train, :-1], data[train, -1]
    x_test, y_test = data[test, :-1], data[test, -1]

    # model = dt.DTLearner(leaf_size=10)
    model = rt.RTLearner(leaf_size=10, random_state=5)

    model.addEvidence(x_train, y_train)

    predictions = model.query(x_test)

    # TODO RTLearner: RT video 2, 46 minutes in

    cv_scores.append(np.sqrt(np.mean((y_test - predictions)**2)))

print np.mean(cv_scores)


