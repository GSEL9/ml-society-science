import pandas as pd
import numpy as np

from sklearn import linear_model
from contextualbandits.online import LinUCB

from .recommender_base import Recommender


class DevRecommender(Recommender):
    """
    Development recommender

    sysout for running test_recommender dev 10000

    ---- Testing with only two treatments ----
    n actions: 2 n_outcomes 2
    Fitting historical data to the policy
    Running an online test
    Testing for  10000 steps
    Total reward: 4506.500000000234
    *** Final analysis of recommender ***
    hehe The policy had a  0.5403 curing rate
    --- Testing with an additional experimental treatment and 126 gene silencing treatments ---
    n actions: 129 n_outcomes 2
    Fitting historical data to the policy
    Running an online test
    Testing for  10000 steps
    Total reward: -747.0000000001357
    *** Final analysis of recommender ***
    hehe The policy had a  0.0253 curing rate
    """
    def __init__(self, n_actions, n_outcomes, explot_after=None):
        # n_actions = min(n_actions, 3)
        super().__init__(n_actions, n_outcomes)

        self.policy = LinUCB(n_actions, n_outcomes)


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
        a = self.policy.predict(user_data)
        assert a.shape[0] == 1
        return a[0]

    def observe(self, user, action, outcome):
        self.policy.partial_fit(user, np.array([action]), np.array([outcome]))
        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])


    def final_analysis(self):
        "Shows which genetic features to look into and a success rate for the treatments"
        # weights = self.policy.coef_
        # gene_weights = weights.ravel()[:128]
        #
        # argmin = gene_weights.argsort()[:3]
        # argmax = gene_weights.argsort()[-3:][::-1]
        #
        # print("Look more into ", [f"gen_{i-1}" for i in argmax], "as they increase likelihood of treatment")
        # print("    as well as ", [f"gen_{i-1}" for i in argmin], "as they decrease likelihood of treatment")

        # treatments = self.observations["action"] == 1
        cured = self.observations["outcome"] == 1
        efficient_treatment = cured

        print("hehe The policy had a ", efficient_treatment.sum()/len(self.observations), "curing rate")
