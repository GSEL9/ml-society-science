import pandas as pd
import numpy as np
from sklearn import linear_model
from recommender import Recommender

class HistoricalRecommender(Recommender):
    """
    The historical recommender approximate the policy pi_0
    """

    def fit_treatment_outcome(self, data: pd.DataFrame,
                                    actions: pd.Series,
                                    outcome: pd.Series):
        """
        Fit a model from patient data, actions and their effects
        Here we assume that the outcome is a direct function of data and actions
        This model can then be used in estimate_utility(), predict_proba() and recommend()
        """
        self._data = data
        self._actions = actions
        self._outcomes = outcome
        self.policy = linear_model.LogisticRegression()
        self.policy.fit(features, actions)


        self.observations = pd.DataFrame({c: [] for c in list(data.columns) + ["action", "outcome"]})


    def recommend(self, user_data):
        y = self.policy.predict(user_data)
        assert y.shape[0] == 1
        return y[0]

    def observe(self, user, action, outcome):
        "We dont care about observing since this policy is not adaptive"
        self.observations.append(np.append(user, [action, outcome]))


    def final_analysis(self):
        print(f"Out of {len(self.observations)} individuals, ")
        self.observations.count()
