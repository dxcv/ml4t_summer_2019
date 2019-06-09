import assess_learners.DTLearner as dt
import assess_learners.RTLearner as rt
import pandas as pd
import numpy as np
from time import time


class BagLearner:

    BAG_SIZE = .60

    def __init__(self, learner=rt.RTLearner, kwargs=None, bags=10, boost=True, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self._model_list = []


    def author(self):
        return 'cfarr31'  # replace tb34 with your Georgia Tech username.

    def addEvidence(self, Xtrain, Ytrain):

        # Total number of train samples
        N = Xtrain.shape[0]

        # Train models
        for _ in range(self.bags):

            # Instantiate model
            model = self.learner(**self.kwargs)

            all_weights = None

            # TODO Should the error be based on the last learner or the ensemble?

            # If boosting and len(model_list) > 0
            if self.boost and len(self._model_list) > 0:
                # rme is the weight, normalize to add to 1
                last_model = self._model_list[-1]
                boost_pred = last_model.query(Xtrain)
                boost_error = (Ytrain - boost_pred) ** 2
                all_weights = boost_error / np.sum(boost_error)

            # Randomly select BAG_SIZE% of the train data
            bag_subset_i = pd.DataFrame(np.arange(0, N)).sample(frac=BagLearner.BAG_SIZE,
                                                                replace=False, weights=all_weights)

            # Re-normalize weight subset
            subset_weights = None
            if self.boost and len(self._model_list) > 0:
                subset_weights = all_weights[bag_subset_i.values.flatten()]
                subset_weights = subset_weights / np.sum(subset_weights)

            # Resample from selection until subset n samples == original n samples
            bag_N_i = bag_subset_i.sample(n=N, replace=True, weights=subset_weights).values.flatten()
            bag_x_train, bag_y_train = Xtrain[bag_N_i], Ytrain[bag_N_i]

            # Train model on subset, append model to list
            model.addEvidence(bag_x_train, bag_y_train)
            self._model_list.append(model)

        return self

    def query(self, Xtest):

        if len(self._model_list) == 0:
            raise AssertionError("Attempting to query an untrained model")

        prediction_list = []
        for model in self._model_list:
            pred = model.query(Xtest)
            prediction_list.append(pred)
        # Average predictions
        prediction_arr = np.stack(prediction_list)
        predictions = prediction_arr.mean(axis=0)

        return predictions

# Bagging: choose 60% of train examples randomly, resample until matching original example n
# Boosting: after each bag is created, test on all of the training data. when selecting the 60%
#   to sample from, weight based on error for choosing



# df = pd.read_csv("assess_learners/Data/Istanbul.csv")
#
#
# data = df.iloc[:, 1:].values
#
#
# # Split all indices into array chunks
# # Implement cross-validation
#
#
# def get_cv_splits(data, k=5, random_state=None):
#     all_indices = np.arange(0, data.shape[0])
#     if random_state is not None:
#         np.random.seed(random_state)
#     else:
#         np.random.seed(np.int(time()))
#     np.random.shuffle(all_indices)
#     test_splits = np.array_split(all_indices, k)
#
#     # Create tuple of train, test indices
#     validation_splits = []
#     for i in range(len(test_splits)):
#         test_i = test_splits[i]
#         train_i = np.concatenate([test_splits[i_] for i_ in range(len(test_splits)) if i_ != i])
#         validation_splits.append((train_i, test_i))
#     return validation_splits
#
#
# cv_scores = []
#
# for train, test in get_cv_splits(data, k=5, random_state=0):
#     # train.shape
#     # test.shape
#     # break
#     # print "Iteration Start...."
#
#     x_train, y_train = data[train, :-1], data[train, -1]
#     x_test, y_test = data[test, :-1], data[test, -1]
#
#     # Create model list
#     model_list = []
#
#     ESTIMATORS = 50
#     LEAF_SIZE = 10
#     BAG_SIZE = .60
#     N = x_train.shape[0]
#     TRAIN_TYPE = "boost"  # bag, boost, normal
#
#     # Train models
#     for _ in range(ESTIMATORS):
#
#         # Instantiate model
#         model = rt.RTLearner(leaf_size=LEAF_SIZE, random_state=_)
#         # model = dt.DTLearner(leaf_size=LEAF_SIZE)
#
#         if TRAIN_TYPE != "normal":
#
#             all_weights = None
#
#             # TODO Should the error be based on the last learner or the ensemble?
#
#             # If boosting and len(model_list) > 0
#             if TRAIN_TYPE == "boost" and len(model_list) > 0:
#                 # rme is the weight, normalize to add to 1
#                 last_model = model_list[-1]
#                 boost_pred = last_model.query(x_train)
#                 boost_error = (y_train - boost_pred)**2
#                 all_weights = boost_error / np.sum(boost_error)
#
#             # TODO Implement bagging
#             # Randomly select 60% of the train data
#             bag_subset_i = pd.DataFrame(np.arange(0, N)).sample(frac=BAG_SIZE, replace=False, weights=all_weights)
#
#             # Re-normalize weight subset
#             subset_weights = None
#             if TRAIN_TYPE == "boost" and len(model_list) > 0:
#                 subset_weights = all_weights[bag_subset_i.values.flatten()]
#                 subset_weights = subset_weights / np.sum(subset_weights)
#
#             # Resample from selection until subset n samples == original n samples
#             bag_N_i = bag_subset_i.sample(n=N, replace=True, weights=subset_weights).values.flatten()
#             bag_x_train, bag_y_train = x_train[bag_N_i], y_train[bag_N_i]
#
#             # Train model on subset, append model to list
#             # Same data (no bagging)
#             model.addEvidence(bag_x_train, bag_y_train)
#
#             # TODO Implement boosting
#             # If no model in list, randomly select 60% of train data
#             # else, test model on train data and generate predictions (just use weights param)
#             #       randomly select 60% of train data with prob weighted by error
#             # Resample from selection until subset n samples == original n samples
#             # Train model on subset, append model to list
#             model_list.append(model)
#         else:
#             model.addEvidence(x_train, y_train)
#             model_list.append(model)
#
#     # Get predictions
#     # print "Generating predictions...."
#     prediction_list = []
#     for model in model_list:
#         pred = model.query(x_test)
#         prediction_list.append(pred)
#     # Average predictions
#     prediction_arr = np.stack(prediction_list)
#     predictions = prediction_arr.mean(axis=0)
#
#     cv_scores.append(np.sqrt(np.mean((y_test - predictions)**2)))
#
# print np.mean(cv_scores)
#

