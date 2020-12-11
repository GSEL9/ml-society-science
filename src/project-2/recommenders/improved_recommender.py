
import numpy as np

from .recommender_base import Recommender
from .policy.min_error_policy import MinErrorPolicy


class ImprovedRecommender(Recommender):
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

        self.policy = MinErrorPolicy()

        # NB: Should be <int>.
        actions = ((actions == 1) & (outcome == 1)).astype(int)

        if actions.ndim == 2 and actions.shape[1] == 1:
            self.policy.fit(data, actions.ravel())
        else:
            self.policy.fit(data, actions)

    def recommend(self, user_data):
        a, = A = self.policy.predict([user_data])
        assert A.shape[0] == 1
        return a

    def observe(self, user, action, outcome):
        "We dont care about observing since this policy is not adaptive"
        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])

    def final_analysis(self):
        "Shows which genetic features to look into and a success rate for the treatments"
    
        treatments = self.observations["action"] == 1
        cured = self.observations["outcome"] == 1
        efficient_treatment = treatments & cured

        print("The policy had a ", cured.sum()/len(self.observations), "curing rate")
