import numpy as np 

from policy import Policy 


class MinErrorPolicy(Policy):
	"""Policy based on the minimum error rate principle.
	"""

	def __init__(self):
		
		self.priors = np.ones(2) * 0.5
		self.L = None 

	def fit(self, data, outcome):
		"""Estimate prior probabilities and likelihoods."""

		# Update priors.
		self.priors[0] = np.sum(outcome == 0) / len(outcome)
		self.priors[1] = np.sum(outcome) / len(outcome)

		if self.L is None:
			self.L = np.ones((2, data.shape[1]))

		# Update likelihood.
		self.L[0] = np.sum(data[outcome == 0], axis=0) / np.sum(data[outcome == 0])
		self.L[1] = np.sum(data[outcome == 1], axis=0) / np.sum(data[outcome == 1])

	def observe(self, data, outcome):
		"""Update prior probabilities and likelihoods."""

		priors_prev = self.priors.copy()
		L_prev = self.L.copy()

		self.fit(data, outcome)

		self.priors = self.priors * priors_prev
		self.L = self.L * L_prev

	def _get_probas(self, x):
		# Discriminant functions.

		C0 = x * np.log(self.L[0] / self.L[1])
		C1 = x * np.log(self.L[1] / self.L[0])

		g1 = np.sum(C0 + np.log(1 - self.L[0])) + np.log(self.priors[0])
		g2 = np.sum(C1 + np.log(1 - self.L[1])) + np.log(self.priors[1])

		p = g1 / (g1 + g2)

		return 1 - p, p

	def get_probas(self, data):
		"""Estimates probailities for each action."""

		return np.array([self._get_probas(x) for x in data], dtype=float)

	def take_action(self, data: np.ndarray, actions: np.ndarray, outcome: np.ndarray, **kwargs):
		"""Select an action."""

		return np.argmax(self.get_probas(data), axis=1)


if __name__ == "__main__":

	import pandas as pd 
	from estimate_utility import expected_utility

	data = pd.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
	actions = pd.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
	outcome = pd.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values

	# NB: Should kill redundant dimension.
	outcome = np.squeeze(outcome)
	
	#policy = ImprovedPolicy()
	#policy.fit(data, actions)

	policy = MinErrorPolicy()
	policy.fit(data, outcome)
	policy_actions = policy.take_action(data, actions, outcome)
	print(np.sum(policy_actions))
	print(expected_utility(data, policy_actions, outcome, policy, return_ci=True))
	policy.observe(data, outcome)
	print(np.sum(policy_actions))
	print(expected_utility(data, policy_actions, outcome, policy, return_ci=True))
	policy.observe(data, outcome)
	print(np.sum(policy_actions))
	print(expected_utility(data, policy_actions, outcome, policy, return_ci=True))
	policy.observe(data, outcome)
	print(np.sum(policy_actions))
	print(expected_utility(data, policy_actions, outcome, policy, return_ci=True))
	