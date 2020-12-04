import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from .policy import Policy


class ImprovedPolicy(Policy):

	def __init__(self, random_state=42, max_iter=100):

		self.random_state = random_state
		self.max_iter = max_iter

		self.classifier = None 

	@property 
	def coef_(self):
		return self.classifier.coef_

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

			grid_search = GridSearchCV(
				estimator=LogisticRegression(random_state=self.random_state,
											 max_iter=self.max_iter,
											 solver="liblinear"), 
				param_grid=param_grid, cv=5
			)
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

	def predict(self, data: np.ndarray, actions: np.ndarray = None, outcome: np.ndarray = None):
		"""Select an action."""

		return self.classifier.predict(data)