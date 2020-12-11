import numpy as np 

from .policy import Policy


class AdaptivePolicy(Policy):
	"""Implementation of DisjointLinUCB."""

	def __init__(self, n_actions, n_outcomes, alpha=1, seed=42):

		self.alpha = alpha
		self.n_actions = n_actions
		self.n_outcomes = n_outcomes

		self.params = None
		self.rnd = np.random.RandomState(seed=seed)	    
		self.action_attempts_ = np.zeros(self.n_actions)

	def predict(self, x, exploit=False):
		"""Predict an action. Ties are handled by randomly selection."""

		if exploit:
		    probas = [self.expected_reward(a, x) for a in range(self.n_actions)]

		else:
		    probas = self.get_probas(x)
		    probas[self.rnd.randint(self.n_actions)] += 1e-6

		action = np.argmax(probas)

		self.action_attempts_[action] += 1

		return action

	def expected_reward(self, a, x):

		theta_hat = np.dot(np.linalg.inv(self.params[a]["A"]), self.params[a]["b"])

		return theta_hat @ x

	def fit(self, data, actions, outcome):
		"""Learn parameters from training data."""

		_, self.n_features = np.shape(data)
		self.params = {a: {'A': np.identity(self.n_features), 'b': np.zeros(self.n_features)} 
	    			   for a in range(self.n_actions)}

	    # Organize data by actions.
		_data = {a: data[actions == a] for a in range(self.n_actions)}
		_outcome = {a: outcome[actions == a] for a in range(self.n_actions)}

		for a, action_params in self.params.items():
			for i, x in enumerate(_data[a]):
		    	
				# Update parameters according to ground truth reward (outcome).
				self.update_params(x, a, _outcome[a][i])

	def update_params(self, x, a, r):  

		self.params[a]["A"] += np.outer(x, x)
		self.params[a]["b"] += r * x

	def get_probas(self, x):

		probas = np.zeros(self.n_actions)
		for a, action_params in self.params.items():

			A_inv = np.linalg.inv(action_params["A"])
			theta_hat = np.dot(A_inv, action_params["b"])

			probas[a] = self.get_proba(x, theta_hat, A_inv)

		return probas

	def get_proba(self, x, theta, A_inv):

		return theta.T @ x + self.alpha * np.sqrt(x.T @ A_inv @ x)

	def partial_fit(self, x, a, y):
		"""Update parameter estiamtes."""

		# New action.
		if a not in range(self.n_actions):	

			self.params[a]["A"] = np.identity(self.n_features)
			self.params[a]["b"] = np.zeros(self.n_features)

		self.update_params(x, a, y)