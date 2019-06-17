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
Student Name: Chris Farr
GT User ID: cfarr31
GT ID: 90347082
"""

import numpy as np
import math


# this function should return a dataset (X and Y) that will work 			  		 			 	 	 		 		 	  		   	  			  	
# better for linear regression than decision trees 			  		 			 	 	 		 		 	  		   	  			  	
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    n_rows = 1000  # from 10 to 1000
    n_cols = 10  # from 2 to 10

    x = np.random.uniform(size=(n_rows, n_cols))

    rando_funcs = [lambda x: x**2, lambda x: np.sqrt(x), lambda x: np.log10(x), lambda x: x**3, lambda x: np.log(x)]

    y_ = x.copy()

    for i in range(y_.shape[1]):
        f = np.random.randint(0, 4)
        y_[:, i] = rando_funcs[f](y_[:, i])

    y = np.sum(x, axis=1)

    return x, y


def best4DT(seed=1489683273):
    np.random.seed(seed)

    n_rows = 1000  # from 10 to 1000
    n_cols = 10  # from 2 to 10

    x = np.random.uniform(-10, 10, (n_rows, n_cols))

    def sub_func(x_):
        y_ = x_.copy()

        # Create random groupings and set y to average within groups
        random_groups = sorted(np.random.uniform(-10, 10, 3))

        y_[x_ < random_groups[0]] = np.mean(y_[x_ < random_groups[0]])

        for i in range(1, len(random_groups)):
            y_[(random_groups[i - 1] < x_) & (x_ < random_groups[i])] = np.mean(
                y_[(random_groups[i - 1] < x_) & (x_ < random_groups[i])])

        y_[x_ > random_groups[-1]] = np.mean(y_[x_ > random_groups[-1]])

        return y_

    y = sub_func(x[:, 0])

    return x, y


def author():
    return 'cfarr31'  # Change this to your user ID
