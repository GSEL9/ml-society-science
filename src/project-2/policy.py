from abc import abstractmethod, ABC
from typing import Tuple, List, Union

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import pandas as pd
import numpy as np

from nn_model import train_ann_model


class Policy(ABC):
	"""
	Abstract base class for a policy.
	"""

	@abstractmethod
	def fit(self, data: np.ndarray, actions: np.ndarray, **kwargs):
		"""Adapt the decision rule to data."""
		raise NotImplementedError("Function fit() not implemented for Policy.")

	@abstractmethod
	def get_probas(self, data: np.ndarray, **kwargs):
		"""Select an action."""

		raise NotImplementedError("Function get_probas() not implemented for Policy.")

	@abstractmethod
	def take_action(self, data: np.ndarray, actions: np.ndarray, outcome: np.ndarray, 
					**kwargs):
		"""Select an action."""

		raise NotImplementedError("Function take_action() not implemented for Policy.")


class ImprovedPolicy(Policy):

	def __init__(self, random_state=42):

		#super().__init__(self)

		self.random_state = random_state
		self.classifier = None 

	def fit(self, data: np.ndarray, actions: np.ndarray, verbose=0, optimize=False, 	
			**kwargs):
		"""Train a model to estimate the probability of an outcome from data.

		Args:
			data: Data matrix.
			actions: Actions for each data sample.
		"""

		if optimize:

			if verbose > 0:
				print("Hyperparameter search...")

			param_grid = [{'C': 10 ** np.linspace(-4, 2, 10)},
						  {'penalty': ['l1', 'l2']}]
			grid_search = GridSearchCV(estimator=LogisticRegression(random_state=42,
																    max_iter=1e4,
																    solver="liblinear"), 
						   			   param_grid=param_grid, cv=5)
			grid_search.fit(data, actions)
			self.classifier = grid_search.best_estimator_

		else:
			self.classifier = LogisticRegression(random_state=self.random_state, 
												 solver="liblinear",
												 **kwargs)
			self.classifier.fit(data, actions)

		return self

	def get_probas(self, data: np.ndarray, **kwargs):
		"""Select an action."""

		return self.classifier.predict_proba(data)

	def take_action(self, data: np.ndarray, actions: np.ndarray, outcome: np.ndarray, 
					**kwargs):
		"""Select an action."""

		if self.classifier is None:
			self.fit(data, np.squeeze(outcome), **kwargs)

		P = self.get_probas(data)

		return 1 - np.greater(outcome * P[:, 0], (outcome - 0.1) * P[:, 1])


class DeepPolicy(Policy):

	def __init__(self, random_state=42):

		self.random_state = random_state
		self.classifier = None

	def fit(self, data: np.ndarray, actions: np.ndarray, verbose=0, optimize=False, 	
			**kwargs):
		"""Train a model to estimate the probability of an outcome from data.

		Args:
			data: Data matrix.
			actions: Actions for each data sample.
		"""

		self.classifier = train_ann_model(data, actions, outcome, **kwargs)

	def get_probas(self, data: np.ndarray, **kwargs):
		"""Select an action."""

		p0 = self.classifier(data).numpy()

		return np.hstack([p0, 1 - p0])

	def take_action(self, data: np.ndarray, actions: np.ndarray, outcome: np.ndarray, 
					**kwargs):
		"""Select an action."""

		if self.classifier is None:
			self.fit(data, np.squeeze(outcome), **kwargs)

		P = self.get_probas(data)
		print(P)
		return 1 - np.greater(outcome * P[:, 0], (outcome - 0.1) * P[:, 1])


if __name__ == "__main__":

	from estimate_utility import expected_utility

	data = pd.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
	actions = pd.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
	outcome = pd.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values

	# NB: Should kill redundant dimension.
	outcome = np.squeeze(outcome)
	
	#policy = ImprovedPolicy()
	#policy.fit(data, actions)

	policy = DeepPolicy()
	policy.fit(data, actions, n_epochs=10, batch_size=1000)
	policy_actions = policy.take_action(data, actions, outcome, verbose=1, optimize=False)
	print(np.sum(policy_actions))

	#print(expected_utility(data, policy_actions, outcome, policy, return_ci=True))
