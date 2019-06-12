""" 			  		 			 	 	 		 		 	  		   	  			  	
template for generating data to fool learners (c) 2016 Tucker Balch 			  		 			 	 	 		 		 	  		   	  			  	
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
 			  		 			 	 	 		 		 	  		   	  			  	
Student Name: Tucker Balch (replace with your name) 			  		 			 	 	 		 		 	  		   	  			  	
GT User ID: tb34 (replace with your User ID) 			  		 			 	 	 		 		 	  		   	  			  	
GT ID: 900897987 (replace with your GT ID) 			  		 			 	 	 		 		 	  		   	  			  	
"""

import numpy as np
import math


# this function should return a dataset (X and Y) that will work 			  		 			 	 	 		 		 	  		   	  			  	
# better for linear regression than decision trees 			  		 			 	 	 		 		 	  		   	  			  	
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    # X = np.zeros((100, 2))
    # Y = np.random.random(size=(100,)) * 200 - 100
    # Here's is an example of creating a Y from randomly generated 			  		 			 	 	 		 		 	  		   	  			  	
    # X with multiple columns 			  		 			 	 	 		 		 	  		   	  			  	
    # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3
    # Start with static dataset size, then move to range (If I have to?)
    n_rows = 1000  # from 10 to 1000
    n_cols = 10  # from 2 to 10
    # Linreg is parametric. Derive the y data using a linear function from the x with some random noise? (optional)
    x = np.random.random(size=(n_rows, n_cols))
    # Generate random parameters
    params = np.random.random(size=(n_cols, ))
    # Create y
    y = np.dot(x, params)

    # Seems to be even better

    # x = np.random.uniform(size=(n_rows, n_cols))
    #
    # rando_funcs = [lambda x: x**2, lambda x: np.sqrt(x), lambda x: np.log10(x), lambda x: x**3, lambda x: np.log(x)]
    #
    # y_ = x.copy()
    #
    # for i in range(y_.shape[1]):
    #     f = np.random.randint(0, 4)
    #     y_[:, i] = rando_funcs[f](y_[:, i])
    #
    # y = np.sum(x, axis=1)

    return x, y


def best4DT(seed=1489683273):
    np.random.seed(seed)
    # X = np.zeros((100, 2))
    # Y = np.random.random(size=(100,)) * 200 - 100

    # DT uses median and selects features based on correlation
    # Create a function that takes randomly generate X data
    # Just pick one or two columns, ensure at least 1 column has no relationship
    # Use columns to map params
    n_rows = 1000  # from 10 to 1000
    n_cols = 10  # from 2 to 10

    # The relationship can't be linear....

    x = np.random.uniform(size=(n_rows, n_cols))

    rando_funcs = [lambda x: x**2, lambda x: np.sqrt(x), lambda x: np.log10(x), lambda x: x**3, lambda x: np.log(x)]

    y_ = x.copy()

    for i in range(y_.shape[1]):
        f = np.random.randint(0, 4)
        y_[:, i] = rando_funcs[f](y_[:, i])

    y = np.sum(x, axis=1)

    return x, y


def author():
    return 'cfarr31'  # Change this to your user ID


if __name__ == "__main__":
    print "they call me Tim."