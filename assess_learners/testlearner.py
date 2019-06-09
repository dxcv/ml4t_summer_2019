""" 			  		 			 	 	 		 		 	  		   	  			  	
Test a learner.  (c) 2015 Tucker Balch 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
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
"""

import numpy as np
import math
import LinRegLearner as lrl
import sys
import time


# def get_cv_splits(data, k=5, random_state=None):
#     all_indices = np.arange(0, data.shape[0], 1)
#     if random_state is not None:
#         np.random.seed(random_state)
#     else:
#         np.random.seed(np.int(time.time()))
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
# df = pd.read_csv("assess_learners/Data/Istanbul.csv")
#
#
# data = df.iloc[:, 1:].values
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
#     BASE_LEARNER = dt.DTLearner
#
#     model = BagLearner(learner=BASE_LEARNER, kwargs={"leaf_size": LEAF_SIZE}, bags=ESTIMATORS, boost=BOOST)
#     model.addEvidence(x_train, y_train)
#     predictions = model.query(x_test)
#
#     cv_scores.append(np.sqrt(np.mean((y_test - predictions)**2)))
#
# print np.mean(cv_scores)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing 			  		 			 	 	 		 		 	  		   	  			  	
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data 			  		 			 	 	 		 		 	  		   	  			  	
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    print testX.shape
    print testY.shape

    # create a learner and train it 			  		 			 	 	 		 		 	  		   	  			  	
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.addEvidence(trainX, trainY)  # train it
    print learner.author()

    # evaluate in sample 			  		 			 	 	 		 		 	  		   	  			  	
    predY = learner.query(trainX)  # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0, 1]

    # evaluate out of sample 			  		 			 	 	 		 		 	  		   	  			  	
    predY = learner.query(testX)  # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0, 1]
