import numpy as np
import pandas as pd
from .recommender_base import Recommender
from .policy.adaptive_policy import AdaptivePolicy


class AdaptiveRecommender(Recommender):
    """
    An adaptive recommender for active treatment. Based on context bandit
    """
    n_actions_factor = 10
    max_exploit_after_n = 2000

    def __init__(self, n_actions, n_outcomes, exploit_after_n=None):

        super().__init__(n_actions, n_outcomes)

        self.policy = AdaptivePolicy(n_actions, n_outcomes)

        self.exploit_after_n = min(self.n_actions_factor * n_actions, self.max_exploit_after_n)

    def fit_treatment_outcome(self, data: np.ndarray,
                                    actions: np.ndarray,
                                    outcome: np.ndarray,
                                    random_state:int=0):
        """
        Fit a model from patient data, actions and their effects
        Here we assume that the outcome is a direct function of data and actions
        This model can then be used in estimate_utility(), predict_proba() and recommend()
        """

        self._data = data
        self._actions = actions
        self._outcomes = outcome

        # NB: Should be <int>.
        actions = ((actions == 1) & (outcome == 1)).astype(int)

        if actions.ndim == 2 and actions.shape[1] == 1:
            self.policy.fit(data, actions.ravel(), outcome.ravel())
        else:
            self.policy.fit(data, actions, outcome)

    def recommend(self, user_data):
        if len(self.observations) == self.exploit_after_n:
            print("STARTING TO EXPLOIT")
        if len(self.observations) > self.exploit_after_n:
            return self.policy.predict(user_data, exploit=True)

        return self.policy.predict(user_data)

    def observe(self, user, action, outcome):
        self.policy.partial_fit(user, action, outcome)
        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])


    def final_analysis(self):
        """
        After all the data has been obtained, do a final analysis. This can consist of a number of things:
        1. Recommending a specific fixed treatment policy
        2. Suggesting looking at specific genes more closely
        3. Showing whether or not the new treatment might be better than the old, and by how much.
        4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
        """
        cured = self.observations["outcome"] == 1
        print("The adaptive policy had a ", cured.sum()/len(self.observations), "curing rate")

        best_fixed_action = self.observations[self.exploit_after_n:][cured]["action"].mode().to_numpy()[0]
        print("1: Recommending fixed policy: action =", best_fixed_action)

        print("2: Look into genes: ", genes)

        print("3: Curing rate for old treatment:", curing_rate_1, "curing rate for new treatment: ", curing_rate_2)

        print("4: ")
