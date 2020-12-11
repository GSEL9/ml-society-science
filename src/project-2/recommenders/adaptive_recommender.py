import numpy as np
import pandas as pd

from .recommender_base import Recommender
from .policy.min_error_policy import MinErrorPolicy


class AdaptiveRecommender(Recommender):
    """
    An adaptive recommender for active treatment. Based on context bandit
    """

    def __init__(self, n_actions, n_outcomes, exploit_after_n=None):
        exploit_after_n = 10*n_actions
        # n_actions = min(n_actions, 3)
        super().__init__(n_actions, n_outcomes)

        self.policy = LinUCB(n_actions, n_outcomes)

        self.exploit_after_n = exploit_after_n

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

        self.policy.fit(data, actions, outcome)

    def recommend(self, user_data):
        if len(self.observations) == self.exploit_after_n:
            print("STARTING TO EXPLOIT")
        if len(self.observations) > self.exploit_after_n:
            a = self.policy.predict(np.array([user_data]), exploit=True)
        else:
            a = self.policy.predict(np.array([user_data]))
        assert a.shape[0] == 1
        return a[0]

    def observe(self, user, action, outcome):
        self.policy.partial_fit(user, np.array([action]), np.array([outcome]))
        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])


    def final_analysis(self):
        "Shows which genetic features to look into and a success rate for the treatments"
        print("The adaptive policy had a ", efficient_treatment.sum()/len(self.observations), "curing rate")

        cured = self.observations["outcome"] == 1

        print("Recommending fixed policy: action ", self.observations[cured]["action"].mode().to_numpy()[0])
        efficient_treatment = cured
