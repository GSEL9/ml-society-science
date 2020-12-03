import numpy as np 

from policy import Policy 


class MinErrorPolicy(Policy):
	"""Policy based on the minimum error rate principle.
	"""

	def __init__(self):
		
		self.priors = np.ones(2) * 0.5
		self.L = None 

	def fit(self, data, outcome):
		"""Train the model."""

		# Update priors.
		self.priors[0] = np.sum(outcome == 0) / len(outcome)
		self.priors[1] = np.sum(outcome) / len(outcome)

		if self.L is None:
			self.L = np.ones((2, data.shape[1]))

		# Update likelihood.
		self.L[0] = np.sum(data[outcome == 0], axis=0) / np.sum(data[outcome == 0])
		self.L[1] = np.sum(data[outcome == 1], axis=0) / np.sum(data[outcome == 1])

	# TODO: Implement updates according to Bayes rule.
	# NOTE: Remeber to scale by sum over all outcomes.
	def update(self, data, outcome):

		raise NotImplementedError("")

	def _get_probas(self, x):

		C0 = x * np.log(self.L[0] / self.L[1])
		C1 = x * np.log(self.L[1] / self.L[0])

		g1 = np.sum(C0 + np.log(1 - self.L[0])) + np.log(self.priors[0])
		g2 = np.sum(C1 + np.log(1 - self.L[1])) + np.log(self.priors[1])

		return 1 - g1 / (g1 + g2), g1 / (g1 + g2)

	def get_probas(self, data):
		"""Discriminant function."""

		return np.array([self._get_probas(x) for x in data], dtype=float)

	def take_action(self, data: np.ndarray, actions: np.ndarray, outcome: np.ndarray, 
					**kwargs):
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
	#print(policy.get_probas(data))
	#print(policy_actions.size, actions.size)
	#print(np.sum(policy_actions == actions) / actions.size)
	print(expected_utility(data, policy_actions, outcome, policy, return_ci=True))
