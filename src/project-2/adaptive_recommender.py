import numpy as np
import pandas as pd
from sklearn import linear_model

from improved_recommender import ImprovedRecommender
cols = ["sex", "smoker"] + [f"gen_{i}" for i in range(128)] + ["action", "outcome"]

class AdaptiveRecommender(ImprovedRecommender):

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
        self.policy = linear_model.SGDClassifier(loss="log", random_state=random_state, max_iter=5000)

        # important: make sure to cast to int. Otherwise, it will not work
        actions = ((actions == 1) & (outcome == 1)).astype(int)

        label_types = np.arange(self.n_actions)

        if actions.ndim == 2 and actions.shape[1] == 1:
            self.policy.partial_fit(data, actions.ravel(), classes=label_types)
        else:
            self.policy.partial_fit(data, actions, classes=label_types)

        self.observations = pd.DataFrame({c: [] for c in cols})

    def observe(self, user, action, outcome):
        target = action & outcome

        # Update model adaptively
        self.policy.partial_fit([user], [target])
        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])
