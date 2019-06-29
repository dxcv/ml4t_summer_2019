""" 			  		 			 	 	 		 		 	  		   	  			  	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch 			  		 			 	 	 		 		 	  		   	  			  	
Note, this is NOT a correct DTLearner; Replace with your own implementation. 			  		 			 	 	 		 		 	  		   	  			  	
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
Student Name: Chris Farr
GT User ID: cfarr31
GT ID: 90347082
"""
import numpy as np


class DTLearner:
    # Indices within self._tree_data array
    NODE_INDEX = 0
    FEAT_INDEX = 1
    SPLIT_VALUE_INDEX = 2
    LEFT_INDEX = 3
    RIGHT_INDEX = 4

    def __init__(self, leaf_size=1, verbose=False):
        self._tree_data = None
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'cfarr31'  # replace tb34 with your Georgia Tech username.

    def addEvidence(self, Xtrain, Ytrain):
        self._tree_data = self.build_tree(Xtrain, Ytrain, leaf_size=self.leaf_size, verbose=self.verbose)
        return self

    @staticmethod
    def build_tree(x, y, depth=0, leaf_size=1, verbose=False):
        # If the x rows are fewer or equal to leaf size, return leaf
        if x.shape[0] <= leaf_size:
            return np.array([[depth, np.nan, np.mean(y), np.nan, np.nan]])
        # If all labels are the same, return leaf
        if np.std(y) == 0:
            return np.array([[depth, np.nan, np.mean(y), np.nan, np.nan]])
        # Calculate absolute correlation between each feature and y
        feature_scores = np.array([np.abs(np.float(np.correlate(x[:, j], y))) for j in range(x.shape[1] - 1)])
        # Features sorted from highest correlation descending
        feature_sort_i = np.argsort(feature_scores)[::-1]
        # Loop through feature_sort_i until find one with std > 0
        # If none are available, return leaf
        feature = np.nan
        for i in feature_sort_i:
            if np.std(x[:, i]) == 0:
                # If zero std in feature split, continue
                continue
            elif np.all(x[:, i] <= np.median(x[:, i])) or not np.any(x[:, i] <= np.median(x[:, i])):
                # If none or all are less than median, continue
                continue
            else:
                feature = i
                break
        if np.isnan(feature):
            return np.array([[depth, np.nan, np.mean(y), np.nan, np.nan]])
        # Split value based on median of selected feature
        split_value = np.median(x[:, feature])
        # Calculate index based on selected feature and split value
        left_index = x[:, feature] <= split_value
        right_index = x[:, feature] > split_value
        # Recursively build tree until all branches have returned a leaf
        left_tree = DTLearner.build_tree(x[left_index, :], y[left_index], depth + 1, leaf_size, verbose)
        right_tree = DTLearner.build_tree(x[right_index, :], y[right_index],
                                          depth + left_tree.shape[0] + 1, leaf_size, verbose)
        # Create root node
        root = np.array([[depth, feature, split_value, depth + 1, depth + left_tree.shape[0] + 1]])
        # Return the combined trees
        return np.vstack([root, left_tree, right_tree])

    def query(self, Xtest):
        if self._tree_data is None:
            raise AssertionError("Attempting to query an untrained model")
        predictions = []
        # Loop through each example in test data
        for i in range(len(Xtest)):
            # Start with node 0
            node = 0
            # Loop to iteratively determine left or right branch on split_value until a leaf is found
            while True:
                # Get the feature for the current node
                feature = np.int(self._tree_data[node, DTLearner.FEAT_INDEX]) if not np.isnan(
                    self._tree_data[node, DTLearner.FEAT_INDEX]) else np.nan
                # Get the split value for the current node
                split_value = self._tree_data[node, DTLearner.SPLIT_VALUE_INDEX]
                # If feature is nan then it's a leaf
                if np.isnan(feature):
                    # Return split_value as the prediction
                    predictions.append(split_value)
                    break
                # If value is less or equal to split value, return node at left index
                if Xtest[i, feature] <= split_value:
                    node = np.int(self._tree_data[node, DTLearner.LEFT_INDEX])
                # Else return node at right index
                else:
                    node = np.int(self._tree_data[node, DTLearner.RIGHT_INDEX])
        return np.array(predictions)
