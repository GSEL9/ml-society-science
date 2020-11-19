from typing import Tuple, List, Union

from scipy import stats

import pandas as pd
import numpy as np 


def confidence_interval(data: Union[List, np.ndarray], alpha: float=0.05) -> Tuple:
	"""Calculate error bounds of normal ditributed data with 1 - alpha confidence. 
	"""

	m, se = np.mean(data), stats.sem(data)
	h = se * stats.t.ppf((2 - alpha) / 2, len(data) - 1)

	return m - h, m + h


def expected_utility(data: np.ndarray, actions: np.ndarray, outcome: np.ndarray, 
					 return_ci=True) -> Union[Tuple, float]: 
	"""Calculate the expected utility from data.

	Args:
		data: Feature matrix.
		actions: Vector of actions.
		outcome: Vector of outcomes.
		return_ci: Calculate a confidence interval for the utility estimate.

	Returns:
		The expected utility with optinal confidence interval.
	"""

	probas = np.asarray([policy.get_probas(x) for x in data])
	probas_a0, probas_a1 = zip(*probas)

	U_expected = sum(outcome * probas_a0 + (outcome - 0.1) * probas_a1)

	# Assuming the probabilities are normal distributed.
	probas_a0_lo, probas_a0_hi = confidence_interval(probas_a0)
	probas_a1_lo, probas_a1_hi = confidence_interval(probas_a1)

	# Sanity check.
	assert np.isclose(probas_a0_lo + probas_a1_hi, 1)
	assert np.isclose(probas_a1_lo + probas_a0_hi, 1)

	# NOTE: U_a0_lo = U_a1_hi and U_a1_lo = U_a0_hi
	U_lo = sum(outcome * probas_a0_lo + (outcome - 0.1) * (1 - probas_a0_lo))
	U_hi = sum(outcome * probas_a0_hi + (outcome - 0.1) * (1 - probas_a0_hi))

	if return_ci:
		return U_lo, U_expected, U_hi

	return U_expected


if __name__ == "__main__":

	features = pd.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
	actions = pd.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
	outcome = pd.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values

	# NB: Should kill redundant dimension.
	outcome = np.squeeze(outcome)	

	expected_utility(features, actions, outcome)