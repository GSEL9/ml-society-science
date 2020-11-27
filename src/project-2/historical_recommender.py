import pandas as pd
import numpy as np
from sklearn import linear_model
from recommender import Recommender

# cols = ["sex", "smoker"] + [f"gen_{i}" for i in range(128)] + ["symptom_1", "symptom_2", "action", "outcome"]
cols = ["sex", "smoker"] + [f"gen_{i}" for i in range(128)] + ["action", "outcome"]

class HistoricalRecommender(Recommender):
    """
    The historical recommender approximate the policy pi_0
    """

    def fit_treatment_outcome(self, data: np.ndarray,
                                    actions: np.ndarray,
                                    outcome: np.ndarray):
        """
        Fit a model from patient data, actions and their effects
        Here we assume that the outcome is a direct function of data and actions
        This model can then be used in estimate_utility(), predict_proba() and recommend()
        """
        self._data = data
        self._actions = actions
        self._outcomes = outcome
        self.policy = linear_model.LogisticRegression(max_iter=5000)
        self.policy.fit(data, actions)

        self.observations = pd.DataFrame({c: [] for c in cols})


    def recommend(self, user_data):
        y = self.policy.predict([user_data])
        assert y.shape[0] == 1
        return y[0]

    def observe(self, user, action, outcome):
        "We dont care about observing since this policy is not adaptive"
        self.observations.append(pd.Series(np.append(user, [action, outcome])), ignore_index=True)


    def final_analysis(self):
        treatments = self.observations["action"] == 1
        cured = self.observations["outcome"] == 1
        efficient_treatment = treatments & cured

        print("The policy had a ", efficient_treatment.sum()/len(self.observations), "curing rate")
