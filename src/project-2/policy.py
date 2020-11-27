from abc import abstractmethod, ABC
from typing import Tuple, List, Union

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import pandas as pd
import numpy as np


class Policy(ABC):
	"""
	Abstract base class for a policy.
	"""

	@abstractmethod
	def fit(self, data: np.ndarray, actions: np.ndarray, **kwargs):
		"""Adapt the decision rule to data."""
		pass

	@abstractmethod
	def take_action(data: np.ndarray, actions: np.ndarray, outcome: np.ndarray, **kwargs):
		"""Select an action."""
		pass


class ImprovedPolicy(Policy):

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

	def take_action(self, data: np.ndarray, actions: np.ndarray, outcome: np.ndarray, 
					**kwargs):

		if self.classifier is None:
			self.fit(data, np.squeeze(outcome), **kwargs)

		P = self.classifier.predict_proba(data)

		return 1 - np.greater(outcome * P[:, 0], (outcome - 0.1) * (1 - P[:, 1])).astype(int)


class DeepPolicy(Policy):

	def __init__(self, epochs, learning_rate, random_state=42):

		self.epochs = epochs
		self.learning_rate = learning_rate
		self.random_state = random_state

		self.classifier = None

	def _initalize(self, X):

		init = tf.random_normal_initializer(seed=self.random_state)

		self.W0 = tf.Variable(init(shape=[X.shape[1], X.shape[0]], dtype=tf.float32))
		self.b0 = tf.Variable(init(shape=[X.shape[0]], dtype=tf.float32))

		self.W1 = tf.Variable(init(shape=[X.shape[0]], dtype=tf.float32))
		self.b1 = tf.Variable(init(shape=[X.shape[0]], dtype=tf.float32))

	# TODO:
	def predict(self):

		pass

	def fit(self, data: np.ndarray, actions: np.ndarray, verbose=0, optimize=False, 	
			**kwargs):

		def loss():

			# Estimate probability using MLP.
			p_a0 = tf.math.sigmoid(tf.nn.relu(X @ self.W0 + self.b0) @ self.W1 + self.b1)
			p_a1 = 1 - p_a0

			return tf.reduce_sum(y * p_a0 + (p_a0 - 0.1) * p_a0)

		self._initalize(data)

		X = tf.cast(data, dtype=tf.float32)
		y = tf.cast(outcome, dtype=tf.float32)

		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
		for _ in range(self.epochs):
			optimizer.minize(self.loss, [decision_variable])

	def take_action(self, data: np.ndarray, actions: np.ndarray, outcome: np.ndarray, 
					**kwargs):

		if self.classifier is None:
			self.fit(data, np.squeeze(outcome), **kwargs)

		P = self.classifier.predict_proba(data)

		return 1 - np.greater(outcome * P[:, 0], (outcome - 0.1) * (1 - P[:, 1])).astype(int)


if __name__ == "__main__":

	data = pd.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
	actions = pd.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
	outcome = pd.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values

	# NB: Should kill redundant dimension.
	outcome = np.squeeze(outcome)
	policy = ImprovedPolicy()
	actions = policy.take_action(data, actions, outcome, verbose=1, optimize=True)
	print(sum(actions))

