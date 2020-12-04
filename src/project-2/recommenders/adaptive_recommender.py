import numpy as np
import pandas as pd

from .improved_recommender import ImprovedRecommender
from .policy.min_error_policy import MinErrorPolicy


cols = ["sex", "smoker"] + [f"gen_{i}" for i in range(128)] + ["action", "outcome"]


class AdaptiveRecommender(ImprovedRecommender):
    """
    An adaptive recommender for active treatment.
    """

    @property 
    def name(self):

        return "AdaptiveRecommender"

    def fit_treatment_outcome(self, data: np.ndarray,
                                    actions: np.ndarray,
                                    outcome: np.ndarray,
                                    random_state:int=0):
        """
        Fit a model from patient data, actions and their effects.
        Here we assume that the outcome is a direct function of data and actions.
        This model can then be used in estimate_utility(), predict_proba() and recommend().
        """

        self._data = data
        self._actions =  np.squeeze(actions)
        self._outcomes = np.squeeze(outcome)
        self.policy = MinErrorPolicy()
        
        # Learn when active treatment will cure an individual to make recommendations.
        idx = np.logical_not(self._actions == 0)
 
        if actions.ndim == 2 and actions.shape[1] == 1:
            self.policy.fit(self._data[idx], self._outcomes[idx].ravel())
        else:
            self.policy.fit(self._data[idx], self._outcomes[idx])

        self.observations = pd.DataFrame({c: [] for c in cols})

    def observe(self, user, action, outcome):
        """Adapt policy parameters to new data."""

        self.policy.observe([user], [outcome])
        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])

    def final_analysis(self):
        """Shows which genetic features to look into and a success rate for the treatments"""

        # Find the genes strongest associated with each outcome.
        gene_probas = self.policy.L[:, 2:128]
        print(f"Most influential gene for cured group: {np.argmax(gene_probas[0])}")
        print(f"Most influential gene for not cured group: {np.argmax(gene_probas[1])}")

        treatments = self.observations["action"] == 1
        cured = self.observations["outcome"] == 1
        efficient_treatment = treatments & cured
        print("The policy had a ", efficient_treatment.sum() / len(self.observations), "curing rate")
