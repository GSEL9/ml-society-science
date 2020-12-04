from abc import abstractmethod, ABC

import numpy as np


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
	def predict(self, data: np.ndarray, **kwargs):
		"""Select an action.

		Kwargs:
			actions:
			outcome: 
		"""

		raise NotImplementedError("Function take_action() not implemented for Policy.")
