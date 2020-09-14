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

	def expected_utility(self):
		"""Calculate the expected utility of a particular action for a given individual.

		Args:
			m: Investment.
			n: Lending_period in months.
		"""

		# Credit worthy.
		if self.predict_proba() > 0.5:
			return m * ((1 + self.rate) ** n - 1)

		return -1.0 * m

	# predicting credit worthiness as input to your policy
	def predict_proba(self):
		"""The probability that a person will return the loan."""
		pass

	# Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
 		"""Create a model estimating probabilities forpredict_proba()."""

    	p_creditworthines = sum(credit_worthy) / total

        self.data = [X, y]
