import numpy as np
import pandas as pd
from .recommender_base import Recommender
from .policy.adaptive_policy import AdaptivePolicy
from contextualbandits.online import LinUCB
# import matplotlib as plt
from collections import Counter

class AdaptiveRecommender(Recommender):
    """
    An adaptive recommender for active treatment. Based on context bandit
    """

    def __init__(self, n_actions, n_outcomes, exploit_after_n=None, n_actions_factor=10, max_exploit_after_n=2000):

        super().__init__(n_actions, n_outcomes)

        # self.policy = AdaptivePolicy(n_actions, n_outcomes)
        self.policy = LinUCB(n_actions)

        self.exploit_after_n = min(n_actions_factor * n_actions, max_exploit_after_n)

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
        x = np.array([user_data])
        if len(self.observations) == self.exploit_after_n:
            print("STARTING TO EXPLOIT")

        if len(self.observations) > self.exploit_after_n:
            a, = self.policy.predict(x, exploit=True)
        else:
            a, = self.policy.predict(x)

        return a

    def observe(self, user, action, outcome):
        x, a, y = (np.array([i]) for i in (user, action, outcome))
        self.policy.partial_fit(x, a, y)
        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])


    def final_analysis(self):
        """
        After all the data has been obtained, do a final analysis. This can consist of a number of things:
        1. Recommending a specific fixed treatment policy
        2. Suggesting looking at specific genes more closely
        3. Showing whether or not the new treatment might be better than the old, and by how much.
        4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
        """
        cured = self.observations["outcome"] == 1
        print("The adaptive policy had a ", cured.sum()/len(self.observations), "curing rate")

        # observation_counts = Counter(self.observations[cured].iloc[self.exploit_after_n:]["action"])
        observation_counts = Counter(self.observations[cured]["action"])

        ocdf = pd.DataFrame(observation_counts, ["counts"])
        ocdf.plot.bar()
        # best_fixed_action = observation_counts.mode().to_numpy()[0]
        # print("1: Recommending fixed policy: action =", best_fixed_action)

        # compute gene imapct
        # mean_abs_thetas = self.policy.mean_magnitude_thetas()
        # print(mean_abs_thetas.shape)
        # gene_weights = mean_abs_thetas[2:128]
        # argmax = gene_weights.argsort()[-3:][::-1]
        #
        # print("2: Look into genes: ", [f"gen_{i-1}" for i in argmax])

        # print("3: Curing rate for old treatment:", curing_rate_1, "curing rate for new treatment: ", curing_rate_2)
        #
        # print("4: ")
