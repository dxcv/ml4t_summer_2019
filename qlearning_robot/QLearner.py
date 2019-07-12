""" 			  		 			 	 	 		 		 	  		   	  			  	
Template for implementing QLearner  (c) 2015 Tucker Balch 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
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
import random as rand


class QLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self._Q = np.zeros(shape=(num_states, num_actions))
        self._experience = list()
        self._alpha = alpha
        self._gamma = gamma
        self._rar = rar
        self._radr = radr
        self._dyna = dyna

    def querysetstate(self, s):
        """ 			  		 			 	 	 		 		 	  		   	  			  	
        @summary: Update the state without updating the Q-table 			  		 			 	 	 		 		 	  		   	  			  	
        @param s: The new state 			  		 			 	 	 		 		 	  		   	  			  	
        @returns: The selected action


        A special version of the query method that sets the state to s, and returns an integer action according to
        the same rules as query() (including choosing a random action sometimes), but it does not execute an update
        to the Q-table. It also does not update rar. There are two main uses for this method: 1) To set the initial
        state, and 2) when using a learned policy, but not updating it.

        """

        # Get updated next action
        action = np.argmax(self._Q[s, :])

        if self.verbose: print "s =", s, "a =", action

        self.a = action
        self.s = s

        # Determine if random or not
        if np.random.random() < self._rar:
            action = rand.randint(0, self.num_actions - 1)

        # Update random score
        self._rar *= self._radr

        return action

    def query(self, s_prime, r):
        """ 			  		 			 	 	 		 		 	  		   	  			  	
        @summary: Update the Q table and return an action 			  		 			 	 	 		 		 	  		   	  			  	
        @param s_prime: The new state 			  		 			 	 	 		 		 	  		   	  			  	
        @param r: The ne state 			  		 			 	 	 		 		 	  		   	  			  	
        @returns: The selected action


        Core method of the Q-Learner.
        Keep track of the last state s and the last action a
        Use the new information s_prime and r to update the Q table
        The learning instance, or experience tuple is <s, a, s_prime, r>.
        return an integer, which is the next action to take. Note that it
        should choose a random action with probability rar
        update rar according to the decay rate radr at each step

        s_prime integer, the the new state.
        r float, a real valued immediate reward.
        """

        # Add experience tuple to agent
        new_experience = (self.s, self.a, s_prime, r)
        self._experience.append(new_experience)

        # Update Q table
        if self.s is not None and self.a is not None:
            self._Q[self.s, self.a] = self.get_q(self.s, self.a, s_prime, r)

        # Get updated next action, just in case the agent learned about the current state
        action = np.argmax(self._Q[s_prime, :])

        if self._dyna > 0:
            self.dyna(self._dyna)

        # Update state and action
        self.s = s_prime
        self.a = action

        # action = rand.randint(0, self.num_actions - 1)
        if self.verbose: print "s =", s_prime, "a =", action, "r =", r

        # Determine if random or not
        if np.random.random() < self._rar:
            action = rand.randint(0, self.num_actions - 1)

        # Update random score
        self._rar *= self._radr

        return action

    def author(self):
        return 'cfarr31'

    def get_q(self, s, a, s_prime, r):
        # If experience at least len 2, updated Q table
        # (1 - alpha) * Q[s, a] + alpha * (r + gamma * Q[s', argmaxa'(Q[s', a'])])
        a_prime = np.argmax(self._Q[s_prime, :])  # Get next action
        orig_r = (1 - self._alpha) * self._Q[s, a]
        revised_r = self._alpha * (r + self._gamma * self._Q[s_prime, a_prime])
        q_prime = orig_r + revised_r
        return q_prime

    def dyna(self, n):
        # Use update_q
        for _ in range(n):
            # Randomly select from previous experience
            s, a, s_prime, r = rand.choice(self._experience)
            self._Q[s, a] = self.get_q(s, a, s_prime, r)
        return
