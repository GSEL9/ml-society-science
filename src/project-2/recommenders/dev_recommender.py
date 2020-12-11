import pandas as pd
import numpy as np

from sklearn import linear_model
from contextualbandits.online import LinUCB

from .recommender_base import Recommender



class DevRecommender(Recommender):
    """
    Development recommender
    """
    def __init__(self, n_actions, n_outcomes):
        super().__init__(n_actions, n_outcomes)

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


    def recommend(self, user_data):
        return 0


    def observe(self, user, action, outcome):

        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])


    def final_analysis(self):
        "Shows which genetic features to look into and a success rate for the treatments"
        cured = self.observations["outcome"] == 1

        print("Recommending fixed policy: action ", self.observations[cured]["action"].mode().to_numpy()[0])
        efficient_treatment = cured

        print("The policy had a ", efficient_treatment.sum()/len(self.observations), "curing rate")
