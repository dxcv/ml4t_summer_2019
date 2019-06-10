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
import assess_learners.RTLearner as rtl
import assess_learners.DTLearner as dtl
import assess_learners.BagLearner as bl
import time
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":

    def get_cv_splits(data, k=5, random_state=None):
        all_indices = np.arange(0, data.shape[0], 1)
        if random_state is not None:
            np.random.seed(random_state)
        else:
            np.random.seed(np.int(time.time()))
        np.random.shuffle(all_indices)
        test_splits = np.array_split(all_indices, k)

        # Create tuple of train, test indices
        validation_splits = []
        for i in range(len(test_splits)):
            test_i = test_splits[i]
            train_i = np.concatenate([test_splits[i_] for i_ in range(len(test_splits)) if i_ != i])
            validation_splits.append((train_i, test_i))
        return validation_splits


    def rmse(actual, pred):
        return np.sqrt(np.mean((actual - pred) ** 2))


    def mae(actual, pred):
        return np.mean(np.abs(actual - pred))


    def max_error(actual, pred):
        return np.max(actual - pred)


    def explained_variance(actual, pred):
        return 1 - (np.var(actual - pred) / np.var(actual))


    # Set function names for dynamic plotting
    rmse.__name__ = "Root Mean Squared Error"
    mae.__name__ = "Mean Absolute Error"
    max_error.__name__ = "Max Error"
    explained_variance.__name__ = "Explained Variance"


    def cv_score(model, data, k=10, scorers=(rmse,), random_state=0):
        train_scores = [[] for _ in range(len(scorers))]
        test_scores = [[] for _ in range(len(scorers))]

        for train, test in get_cv_splits(data, k=k, random_state=random_state):
            x_train, y_train = data[train, :-1], data[train, -1]
            x_test, y_test = data[test, :-1], data[test, -1]

            # Instantiate and train model
            model.addEvidence(x_train, y_train)

            # Train prediction
            train_pred = model.query(x_train)
            for i, score_list in enumerate(train_scores):
                scorer = scorers[i]
                score_list.append(scorer(y_train, train_pred))

            # Test prediction
            test_pred = model.query(x_test)
            for i, score_list in enumerate(test_scores):
                scorer = scorers[i]
                score_list.append(scorer(y_test, test_pred))

        return [np.mean(score_list) for score_list in train_scores], [np.mean(score_list) for score_list in test_scores]


    # Load data once for all plots
    df = pd.read_csv("Data/Istanbul.csv")

    data = df.iloc[:, 1:].values

    """Create a report that addresses the following questions. Use 11pt font and single spaced lines. We expect that a 
    complete report addressing all the criteria would be at least 3 pages. It should be no longer than 3000 words. To 
    encourage conciseness we will deduct 10 points if the report is too long. The report should be submitted as 
    report.pdf in PDF format. Include charts (not tables) to support each of your answers. 
    
        Does overfitting occur with respect to leaf_size? Use the dataset istanbul.csv with DTLearner. For which values 
        of leaf_size does overfitting occur? Use RMSE as your metric for assessing overfitting. Support your assertion 
        with graphs/charts. (Don't use bagging). 
    
        Can bagging reduce or eliminate overfitting with respect to leaf_size? Again use the dataset istanbul.csv with 
        DTLearner. To investigate this choose a fixed number of bags to use and vary leaf_size to evaluate. Provide 
        charts to validate your conclusions. Use RMSE as your metric. 
    
        Quantitatively compare "classic" decision trees (DTLearner) versus random trees (RTLearner). In which ways is one 
        method better than the other? Provide at least two quantitative measures. Important, using two similar measures 
        that illustrate the same broader metric does not count as two. (For example, do not use two measures for 
        accuracy.) Note for this part of the report you must conduct new experiments, don't use the results of the 
        experiments above for this. 
    
    Note that all charts you provide must be generated in Python (in testlearner.py), and you must submit the python code 
    you used to generate the plots. 
    
    """

    """
    Does overfitting occur with respect to leaf_size? Use the dataset istanbul.csv with DTLearner. For which values 
    of leaf_size does overfitting occur? Use RMSE as your metric for assessing overfitting. Support your assertion 
    with graphs/charts. (Don't use bagging).
    
    
    Reaction: Overfitting occurs with a leaf-size of less than 7 in this experiment. This is evident because the train error is
    relatively low while the test error is relatively high and declining until it reaches the global minimum with a leaf size of 7. 
    As the leaf size increases after this, the model begins to underfit and the test error increases again. 
    
    This trend makes sense because smaller leaf nodes tend to learn a training dataset exactly, while larger leaf nodes
    tend to learn generalizations about the training data that are better at predicting unseen data.
    
    The train and test scores were evaluated using RMSE and 10-fold cross validation.
    
    """

    cv_train_scores = []
    cv_test_scores = []

    leaf_range = range(1, 50, 1)
    for i in leaf_range:
        LEAF_SIZE = i
        model = dtl.DTLearner(leaf_size=LEAF_SIZE)
        train_error, test_error = cv_score(model, data, k=10, random_state=0)
        cv_train_scores.append(train_error[0])
        cv_test_scores.append(test_error[0])

    # Use cv
    # Loop through leaf sizes from 1 to 100 (or some other arbitrary point)
    # Plot average train error and average test error for each split
    # RMSE for error

    plot_df = pd.DataFrame(data=zip(cv_train_scores, cv_test_scores), columns=["Train Error", "Test Error"])
    plot_df.plot(color=["blue", "orange"])
    min_test_error_loc = np.argmin(plot_df["Test Error"])
    min_test_error = plot_df["Test Error"].min()
    plt.axvline(x=min_test_error_loc, color="navajowhite")
    plt.text(min_test_error_loc + .5, min_test_error + .0005,
             "Minimum Test Error: %.4f\nLeaf Size: %s" % (min_test_error, min_test_error_loc),
             verticalalignment='center')

    plt.title("DTLearner - Overfitting Against Leaf Size")
    plt.ylabel("Root Mean Squared Error")
    plt.xlabel("Leaf Size")
    plt.savefig("leaf_size_and_over_fitting.png")
    plt.close()
    import sys
    sys.exit(0)

    """
    Can bagging reduce or eliminate overfitting with respect to leaf_size? Again use the dataset istanbul.csv with 
    DTLearner. To investigate this choose a fixed number of bags to use and vary leaf_size to evaluate. Provide 
    charts to validate your conclusions. Use RMSE as your metric. 
    
    
    When bagging, leaf-size no longer helps to reduce overfitting based on this analysis. At any point, a larger leaf
    size reduces the ability for the model to fit the data for both in sample, and out of sample errors (train and test 
    errors respectively). The optimal leaf-size when using bagging is 1, and the optimal test error is reduced from .0073
    to .0044.
    
    In my implementation I followed the general rule of thumb of each bag to contain roughly 60% of the training data, and 
    then to randomly resample from this subset until the same number of examples exist. This means that even with a 
    leaf size of 1, a single learner still cannot possibly memorize the training data, and this is how overfitting is 
    prevented in this case.
    
    """

    scorers = (rmse,)
    # bags = 5
    bags = 10

    cv_train_scores = []
    cv_test_scores = []

    leaf_range = range(1, 15, 1)
    for i in leaf_range:
        LEAF_SIZE = i
        model = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": LEAF_SIZE}, bags=bags, boost=False)
        train_error, test_error = cv_score(model, data, k=10, scorers=scorers)
        cv_train_scores.append(train_error)
        cv_test_scores.append(test_error)

    # Use cv
    # Loop through leaf sizes from 1 to 100 (or some other arbitrary point)
    # Plot average train error and average test error for each split
    # RMSE for error

    score_arr = np.squeeze(np.array(zip(cv_train_scores, cv_test_scores)))

    plot_df = pd.DataFrame(data=score_arr, columns=["Train Error", "Test Error"],
                           index=leaf_range)
    plot_df.plot(color=["blue", "orange"])
    min_test_error_loc = np.argmin(plot_df["Test Error"])
    min_test_error = plot_df["Test Error"].min()
    plt.axvline(x=min_test_error_loc, color="green")
    plt.text(min_test_error_loc + .5, min_test_error + .0007,
             "Min Test Error:\n %.4f\nLeaf Size: \n%s" % (min_test_error, min_test_error_loc),
             verticalalignment='center')
    plt.title("DTLearner with Bagging - %s Bags \n Overfitting Against Leaf Size" % bags)
    plt.ylabel(scorers[0].__name__)
    plt.xlabel("Leaf Size")
    plt.savefig("bagging_and_over_fitting_%s_bags.png" % bags)
    plt.close()

    """
    Quantitatively compare "classic" decision trees (DTLearner) versus random trees (RTLearner). In which ways is one 
    method better than the other? Provide at least two quantitative measures. Important, using two similar measures 
    that illustrate the same broader metric does not count as two. (For example, do not use two measures for 
    accuracy.) Note for this part of the report you must conduct new experiments, don't use the results of the 
    experiments above for this. 
    
    Experiments:
    Decision Tree vs Random Tree with varying leaf sizes
        Compare rmse to give context compared to previous experiments
        Max error
        Explained Variance
        
    
    
    For nearly any performance test, a single decision tree outperforms a single random tree model. The only exception
    is likely to be computation time, however, in my experiments I was not able to capture a significant difference in the 
    instanbul dataset. 
    
    RTLearner 
    
    
    """

    # Compare DTLearner Test Score to RTLearner Test Scores
    #   Scores:
    #       Max error
    #       MAE
    #       RMSE
    #       Calculation Time
    # Then pick 2 final to keep within single plot

    scorers = (rmse, mae, max_error, explained_variance)

    dt_scores = []
    rt_scores = []

    leaf_range = range(1, 75, 1)
    for i in leaf_range:
        LEAF_SIZE = i

        # Train and test DTLearner
        model = dtl.DTLearner(leaf_size=LEAF_SIZE)
        _, test_error = cv_score(model, data, k=10, random_state=0, scorers=scorers)
        dt_scores.append(test_error)

        # Train and test RTLearner
        model = rtl.RTLearner(leaf_size=LEAF_SIZE, random_state=2019)
        _, test_error = cv_score(model, data, k=10, random_state=0, scorers=scorers)
        rt_scores.append(test_error)

    score_arr = np.concatenate([np.array(dt_scores), np.array(rt_scores)], axis=1)
    col_names = ["DT " + s.__name__ for s in scorers] + ["RT " + s.__name__ for s in scorers]

    plot_df = pd.DataFrame(data=score_arr,
                           columns=col_names,
                           index=leaf_range)

    # Plot RMSE
    score_i_dt = 0
    score_i_rt = score_i_dt + 4
    plot_df.iloc[:, [score_i_dt, score_i_rt]].plot(color=["blue", "orange"])

    dt_loc = np.argmin(plot_df.iloc[:, score_i_dt])
    dt_score = plot_df.iloc[:, score_i_dt].min()

    rt_loc = np.argmin(plot_df.iloc[:, score_i_rt])
    rt_score = plot_df.iloc[:, score_i_rt].min()

    y_offset_dt = dt_score * 1.05
    y_offset_rt = rt_score * 1.05

    plt.axvline(x=dt_loc, color="skyblue")
    plt.text(dt_loc + 1, y_offset_dt,
             "Best DT: %.4f\nLeaf Size: %s" % (dt_score, dt_loc),
             verticalalignment='center')

    plt.axvline(x=rt_loc, color="navajowhite")
    plt.text(rt_loc + 1, y_offset_rt,
             "Best RT: %.4f\nLeaf Size: %s" % (rt_score, rt_loc),
             verticalalignment='center')

    plt.title("Decision Tree (DT) vs Random Tree (RT)\n%s" % scorers[score_i_dt].__name__)
    plt.ylabel(scorers[score_i_dt].__name__)
    plt.xlabel("Leaf Size")
    plt.savefig("dtlearner_vs_rtlearner_%s" % scorers[score_i_dt].__name__)
    plt.close()

    # Plot Mean Absolute Error
    score_i_dt = 1
    score_i_rt = score_i_dt + 4
    plot_df.iloc[:, [score_i_dt, score_i_rt]].plot(color=["blue", "orange"])

    dt_loc = np.argmin(plot_df.iloc[:, score_i_dt])
    dt_score = plot_df.iloc[:, score_i_dt].min()

    rt_loc = np.argmin(plot_df.iloc[:, score_i_rt])
    rt_score = plot_df.iloc[:, score_i_rt].min()

    y_offset_dt = dt_score * 1.05
    y_offset_rt = rt_score * 1.05

    plt.axvline(x=dt_loc, color="skyblue")
    plt.text(dt_loc + 1, y_offset_dt,
             "Best DT: %.4f\nLeaf Size: %s" % (dt_score, dt_loc),
             verticalalignment='center')

    plt.axvline(x=rt_loc, color="navajowhite")
    plt.text(rt_loc + 1, y_offset_rt,
             "Best RT: %.4f\nLeaf Size: %s" % (rt_score, rt_loc),
             verticalalignment='center')

    plt.title("Decision Tree (DT) vs Random Tree (RT)\n%s" % scorers[score_i_dt].__name__)
    plt.ylabel(scorers[score_i_dt].__name__)
    plt.xlabel("Leaf Size")
    plt.savefig("dtlearner_vs_rtlearner_%s" % scorers[score_i_dt].__name__)
    plt.close()

    # Plot Max Error

    score_i_dt = 2
    score_i_rt = score_i_dt + 4
    plot_df.iloc[:, [score_i_dt, score_i_rt]].plot(color=["blue", "orange"])

    dt_loc = np.argmin(plot_df.iloc[:, score_i_dt])
    dt_score = plot_df.iloc[:, score_i_dt].min()

    rt_loc = np.argmin(plot_df.iloc[:, score_i_rt])
    rt_score = plot_df.iloc[:, score_i_rt].min()

    y_offset_dt = dt_score * 1.05
    y_offset_rt = rt_score * 1.10

    plt.axvline(x=dt_loc, color="skyblue")
    plt.text(dt_loc + 1, y_offset_dt,
             "Best DT: %.4f\nLeaf Size: %s" % (dt_score, dt_loc),
             verticalalignment='center')

    plt.axvline(x=rt_loc, color="navajowhite")
    plt.text(rt_loc + 1, y_offset_rt,
             "Best RT: %.4f\nLeaf Size: %s" % (rt_score, rt_loc),
             verticalalignment='center')

    plt.title("Decision Tree (DT) vs Random Tree (RT)\n%s" % scorers[score_i_dt].__name__)
    plt.ylabel(scorers[score_i_dt].__name__)
    plt.xlabel("Leaf Size")
    plt.savefig("dtlearner_vs_rtlearner_%s" % scorers[score_i_dt].__name__)
    plt.close()

    # Plot Max Error

    score_i_dt = 3
    score_i_rt = score_i_dt + 4
    plot_df.iloc[:, [score_i_dt, score_i_rt]].plot(color=["blue", "orange"])

    dt_loc = np.argmax(plot_df.iloc[:, score_i_dt])
    dt_score = plot_df.iloc[:, score_i_dt].max()

    rt_loc = np.argmax(plot_df.iloc[:, score_i_rt])
    rt_score = plot_df.iloc[:, score_i_rt].max()

    y_offset_dt = dt_score * .65
    y_offset_rt = rt_score * .65

    plt.axvline(x=dt_loc, color="skyblue")
    plt.text(dt_loc + 1, y_offset_dt,
             "Best DT: %.4f\nLeaf Size: %s" % (dt_score, dt_loc),
             verticalalignment='center')

    plt.axvline(x=rt_loc, color="navajowhite")
    plt.text(rt_loc + 1, y_offset_rt,
             "Best RT: %.4f\nLeaf Size: %s" % (rt_score, rt_loc),
             verticalalignment='center')

    plt.title("Decision Tree (DT) vs Random Tree (RT)\n%s" % scorers[score_i_dt].__name__)
    plt.ylabel(scorers[score_i_dt].__name__)
    plt.xlabel("Leaf Size")
    plt.ylim((0, None))
    plt.savefig("dtlearner_vs_rtlearner_%s" % scorers[score_i_dt].__name__)
    plt.close()

    # TODO Decide on adding grid after done with plots

    # if __name__ == "__main__":
    #     if len(sys.argv) != 2:
    #         print "Usage: python testlearner.py <filename>"
    #         sys.exit(1)
    #     inf = open(sys.argv[1])
    #     data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    #
    #     # compute how much of the data is training and testing
    #     train_rows = int(0.6 * data.shape[0])
    #     test_rows = data.shape[0] - train_rows
    #
    #     # separate out training and testing data
    #     trainX = data[:train_rows, 0:-1]
    #     trainY = data[:train_rows, -1]
    #     testX = data[train_rows:, 0:-1]
    #     testY = data[train_rows:, -1]
    #
    #     print testX.shape
    #     print testY.shape
    #
    #     # create a learner and train it
    #     learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    #     learner.addEvidence(trainX, trainY)  # train it
    #     print learner.author()
    #
    #     # evaluate in sample
    #     predY = learner.query(trainX)  # get the predictions
    #     rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
    #     print
    #     print "In sample results"
    #     print "RMSE: ", rmse
    #     c = np.corrcoef(predY, y=trainY)
    #     print "corr: ", c[0, 1]
    #
    #     # evaluate out of sample
    #     predY = learner.query(testX)  # get the predictions
    #     rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    #     print
    #     print "Out of sample results"
    #     print "RMSE: ", rmse
    #     c = np.corrcoef(predY, y=testY)
    #     print "corr: ", c[0, 1]
