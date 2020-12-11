import numpy as np
import pandas as pd
from .recommender_base import Recommender
from .policy.adaptive_policy import AdaptivePolicy


class AdaptiveRecommender(Recommender):
    """
    An adaptive recommender for active treatment. Based on context bandit
    """

    def __init__(self, n_actions, n_outcomes, exploit_after_n=None):

        super().__init__(n_actions, n_outcomes)

        self.policy = AdaptivePolicy(n_actions, n_outcomes)

        self.exploit_after_n = min(10 * n_actions, 50)

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
        "Shows which genetic features to look into and a success rate for the treatments"
        cured = self.observations["outcome"] == 1
        print("The adaptive policy had a ", cured.sum()/len(self.observations), "curing rate")


        print("Recommending fixed policy: action ", self.observations[cured]["action"].mode().to_numpy()[0])
        efficient_treatment = cured
