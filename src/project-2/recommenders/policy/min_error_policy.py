import numpy as np 

from .policy import Policy


class MinErrorPolicy(Policy):
	"""Policy based on the minimum error rate principle.
	"""

	def __init__(self):
		
		self.priors = np.ones(2) * 0.5
		self.P = None

	def fit(self, data, actions):
		"""Estimate prior probabilities and likelihoods."""

		# Update priors.
		self.priors[0] = np.sum(actions == 0) / len(actions)
		self.priors[1] = np.sum(actions) / len(actions)

		if self.P is None:
			self.P = np.ones((2, data.shape[1]))

		# Update likelihood.
		self.P[0] = np.sum(data[actions == 0], axis=0) / np.sum(data[actions == 0])
		self.P[1] = np.sum(data[actions == 1], axis=0) / np.sum(data[actions == 1])

	def observe(self, data, actions):
		"""Update prior probabilities and likelihoods."""

		priors_prev = self.priors.copy()
		P_prev = self.P.copy()
	
		self.fit(data, actions)

		self.priors = self.priors + priors_prev
		self.priors = self.priors / sum(self.priors)
			
		self.P = self.P + P_prev
		self.P[0] = self.P[0] / np.sum(self.P[0])
		self.P[1] = self.P[1] / np.sum(self.P[1])
		
	def _get_probas(self, x):
		# Discriminant functions.

		P0 = x * np.log(self.P[0] / self.P[1]) + np.log(1 - self.P[0])
		P1 = x * np.log(self.P[1] / self.P[0]) + np.log(1 - self.P[0])

		g0 = np.sum(P0) + np.log(self.priors[0] + 1e-12)
		g1 = np.sum(P1) + np.log(self.priors[1] + 1e-12)

		p = g0 / (g0 + g1)

		return 1 - p, p

	def get_probas(self, data):
		"""Estimates probailities for each action."""

		return np.array([self._get_probas(x) for x in data], dtype=float)

	def predict(self, data: np.ndarray, **kwargs):
		"""Select an action."""

		return np.argmax(self.get_probas(data), axis=1)