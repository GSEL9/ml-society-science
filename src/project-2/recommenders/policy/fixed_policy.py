import numpy as np 

from .policy import Policy


class FixedPolicy(Policy):
	"""Fixed treatment policy.
	"""

	def __init__(self, treatment=0):
		
		self.treatment = treatment

	def get_probas(self):

		probas = np.zeros(2, dtype=float)
		probas[self.treatment] = 1

		return probas

	def predict(self):

		return np.argmax(self.get_probas())
