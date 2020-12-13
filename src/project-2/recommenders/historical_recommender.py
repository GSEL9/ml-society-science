import pandas as pd
import numpy as np
from sklearn import linear_model

from .recommender_base import Recommender


class HistoricalRecommender(Recommender):
    """
    The historical recommender approximate the policy pi_0
    """

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

        self.policy = linear_model.LogisticRegression(random_state=random_state, max_iter=5000)

        if actions.ndim == 2 and actions.shape[1] == 1:
            self.policy.fit(data, actions.ravel())
        else:
            self.policy.fit(data, actions)

    def recommend(self, user_data):
        """Recommends an action based on approximated historical policy"""
        a = self.policy.predict([user_data])
        assert a.shape[0] == 1
        return a[0]

    def observe(self, user, action, outcome):
        "We dont care about observing since this policy is not adaptive. However, we keep track of the data as we store it for future usage"
        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])


    def final_analysis(self):
        "Shows which genetic features to look into and a success rate for the treatments"
        weights = self.policy.coef_
        gene_weights = weights.ravel()[:128]

        argmin = gene_weights.argsort()[:3]
        argmax = gene_weights.argsort()[-3:][::-1]

        print("Look more into ", [f"gen_{i-1}" for i in argmax], "as they increase likelihood of treatment")
        print("    as well as ", [f"gen_{i-1}" for i in argmin], "as they decrease likelihood of treatment")

        treatments = self.observations["action"] == 1
        cured = self.observations["outcome"] == 1
        efficient_treatment = treatments & cured

        print("The policy had a ", cured.sum()/len(self.observations), "curing rate")
