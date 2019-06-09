import pandas as pd
import numpy as np


class BagLearner:

    # Percent of examples to randomly sample per bag
    BAG_SIZE = .60

    def __init__(self, learner=None, kwargs=None, bags=10, boost=True, verbose=False):
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
            # If using boosting, this will be replaced with error-based weights, else None is passed to sampling weights
            all_weights = None
            # Should the error be based on the last learner or the ensemble?
            # If boosting and len(model_list) > 0
            if self.boost and len(self._model_list) > 0:
                # Get the training error of entire train set from last trained model
                # last_model = self._model_list[-1]
                # boost_pred = last_model.query(Xtrain)
                boost_pred = self.query(Xtrain)
                # Squared error to penalize larger errors
                boost_error = (Ytrain - boost_pred) ** 2  # Add small error to avoid non-zero errors
                boost_error = boost_error + .000001
                # Normalize weights to sum to 1
                all_weights = boost_error / np.sum(boost_error)
            # Randomly select BAG_SIZE% of the train data
            bag_subset_i = pd.DataFrame(np.arange(0, N)).sample(frac=BagLearner.BAG_SIZE,
                                                                replace=False, weights=all_weights)

            # Re-normalize weight subset
            # If boosting, this will be replaced with error-based weights for the sample
            subset_weights = None
            if self.boost and len(self._model_list) > 0:
                # Get weights for selected features
                subset_weights = all_weights[bag_subset_i.values.flatten()]
                # Normalize weights to sum to 1
                subset_weights = subset_weights / np.sum(subset_weights)
            # Re-sample from selection until subset n samples == original n samples
            bag_i = bag_subset_i.sample(n=N, replace=True, weights=subset_weights).values.flatten()
            bag_x_train, bag_y_train = Xtrain[bag_i], Ytrain[bag_i]
            # Train model on subset, append model to list
            model.addEvidence(bag_x_train, bag_y_train)
            self._model_list.append(model)

        return self

    def query(self, Xtest):
        if len(self._model_list) == 0:
            raise AssertionError("Attempting to query an untrained model")
        prediction_arr = np.stack([model.query(Xtest) for model in self._model_list])
        return prediction_arr.mean(axis=0)
