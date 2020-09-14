import numpy as np 

class NameBanker:
	"""
	Args:
		r: Interest rate. 
	"""

    def set_interest_rate(self, rate):
        self.rate = rate

    def get_best_action(self):
    	"""Return the action maximising expected utility."""
    	pass

	def expected_utility(self, P, action, U):
		"""Calculate the expected utility of a particular action for a given individual.
		"""

	    _, n_outcomes = np.shape(P)
	    n_actions, _ = np.shape(U)

	    utility = 0
	    for a in range(n_actions):
	        for w in range(n_outcomes):

	            P_w = get_proba(P, w)
	            utility += U[action, w] * P_w

	    return utility

	def get_proba(self, P, w):

		return 0.4

	# predicting credit worthiness as input to your policy
	def predict_proba(self):
		"""The probability that a person will return the loan."""
			
		# NB: Random seed. 
		p_credit_worthy = np.random.choice(self.posterior)

	# Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    # for giving or denying credit to individuals
    def fit(self, X, y):
 		"""Create a model estimating probabilities for predict_proba()."""

        self.data = [X, y]

        likelihood = ...
        prior = ...
        self.posterior = likelihood * prior


if __name__ == "__main__":

	# TODO: Put whatever goes here in notebook.
	n_models = 10
	n_outcomes = 2
	interest_rate = 0.017

	P = np.zeros([n_models, n_outcomes])
	prior = np.ones(n_models) / n_models

	decision_maker = NameBanker()
	decision_maker.set_interest_rate(interest_rate)

	#for model in range(n_models):
		# Likelihood form data.
	#	P[model, 1] = 
	#	P[model, 0] = 
	#p_credit_worthy = name_banker.predict_proba()

	#action = name_banker.get_best_action(belief, P, U)
	#belief = name_banker.get_posterior(belief, P, ...)
