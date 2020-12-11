
import numpy as np

from .recommender_base import Recommender
from .policy.fixed_policy import FixedPolicy


class FixedRecommender(Recommender):
    """
    The historical recommender approximate the policy pi_0
    """

    def fit_treatment_outcome(self, treatment, *args, **kwargs):
        """
        Fit a model from patient data, actions and their effects
        Here we assume that the outcome is a direct function of data and actions
        This model can then be used in estimate_utility(), predict_proba() and recommend()
        """

        self.policy = FixedPolicy(treatment=treatment)

    def recommend(self, user_data):
        a, = A = self.policy.predict()
        assert A.shape[0] == 1
        return a

    def observe(self, user, action, outcome):
        "We dont care about observing since this policy is not adaptive"
        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])

    def final_analysis(self):
        "Shows which genetic features to look into and a success rate for the treatments"
        # weights = self.policy.coef_
        # gene_weights = weights.ravel()[:128]
        # argmin = gene_weights.argsort()[:3]
        # argmax = gene_weights.argsort()[-3:][::-1]
        #
        # print("Look more into ", [f"gen_{i-1}" for i in argmax], "as they increase likelihood of treatment")
        # print("    as well as ", [f"gen_{i-1}" for i in argmin], "as they decrease likelihood of treatment")

        treatments = self.observations["action"] == 1
        cured = self.observations["outcome"] == 1
        efficient_treatment = treatments & cured

        print("The policy had a ", cured.sum()/len(self.observations), "curing rate")
