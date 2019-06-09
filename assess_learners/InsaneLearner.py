# InsaneLearner should contain 20 BagLearner instances where each instance is composed of 20 LinRegLearner instances
import assess_learners.BagLearner as bl
import assess_learners.LinRegLearner as lrl
import numpy as np


class InsaneLearner:

    N_LEARNERS = 20
    BAGS_PER_LEARNER = 20

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._model_list = []

    def author(self):
        return 'cfarr31'  # replace tb34 with your Georgia Tech username.

    def addEvidence(self, Xtrain, Ytrain):
        for _ in range(InsaneLearner.N_LEARNERS):
            model = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=InsaneLearner.BAGS_PER_LEARNER, boost=False)
            self._model_list.append(model.addEvidence(Xtrain, Ytrain))
        return self

    def query(self, Xtest):
        prediction_arr = np.stack([model.query(Xtest) for model in self._model_list])
        return prediction_arr.mean(axis=0)

#
# import pandas as pd
# from time import time
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
#     ESTIMATORS = 50
#     LEAF_SIZE = 10
#     BOOST = True
#     # BASE_LEARNER = rt.RTLearner
#     # BASE_LEARNER = dt.DTLearner
#
#     # model = BagLearner(learner=BASE_LEARNER, kwargs={"leaf_size": LEAF_SIZE}, bags=ESTIMATORS, boost=BOOST)
#     model = InsaneLearner()
#     model.addEvidence(x_train, y_train)
#     predictions = model.query(x_test)
#
#     cv_scores.append(np.sqrt(np.mean((y_test - predictions)**2)))
#
# print np.mean(cv_scores)
#


