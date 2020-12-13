import numpy as np
import pandas as pd
from .recommender_base import Recommender
from .policy.adaptive_policy import AdaptivePolicy
from contextualbandits.online import LinUCB
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

class AdaptiveRecommender(Recommender):
    """
    An adaptive recommender for active treatment. Based on context bandit
    """

    def __init__(self, n_actions, n_outcomes, exploit_after_n=None, n_actions_factor=20, max_exploit_after_n=2500, n_features=50):

        super().__init__(n_actions, n_outcomes)

        # self.policy = AdaptivePolicy(n_actions, n_outcomes)
        self.policy = LinUCB(n_actions)

        self.exploit_after_n = min(n_actions_factor * n_actions, max_exploit_after_n)
        self.n_features = n_features

    def fit_data(self) -> None:
        if self.n_features is None:
            self.indices = np.arange(self._data.shape[1])

        else:
            importance = []

            kfolds = KFold(5, shuffle=True, random_state=42)
            for train_idx, test_idx in kfolds.split(self._data):

                clf = RandomForestClassifier(n_estimators=100)
                clf.fit(self._data[train_idx], self._outcomes[train_idx].ravel())

                importance.append(clf.feature_importances_)

            avg_importance = np.mean(importance, axis=0)
            idx = np.argsort(avg_importance)[::-1]
            self.indices = idx[:self.n_features]


    def fit_treatment_outcome(self, data: np.ndarray,
                                    actions: np.ndarray,
                                    outcome: np.ndarray,
                                    random_state:int=0) -> None:
        """
        Fit a model from patient data, actions and their effects
        Here we assume that the outcome is a direct function of data and actions
        This model can then be used in estimate_utility(), predict_proba() and recommend()
        """

        self._data = data
        self._actions = actions
        self._outcomes = outcome

        self.fit_data()
        data = data[:, self.indices]

        print(data.shape)


        # NB: Should be <int>.
        actions = ((actions == 1) & (outcome == 1)).astype(int)

        if actions.ndim == 2 and actions.shape[1] == 1:
            self.policy.fit(data, actions.ravel(), outcome.ravel())
        else:
            self.policy.fit(data, actions, outcome)

    def recommend(self, user_data: np.ndarray) -> int:
        x = np.array([user_data[self.indices]])

        if len(self.observations) == self.exploit_after_n:
            print("STARTING TO EXPLOIT")

        if len(self.observations) > self.exploit_after_n:
            a, = self.policy.predict(x, exploit=True)
        else:
            # a, = self.policy.predict(x)
            A = self.policy.topN(x, self.n_actions)
            a = A[(A < 3) | ((A >= 3) & (user_data[A-1] == 1))][-1]

        return a


    def observe(self, user: np.ndarray, action: int, outcome: int) -> None:
        x, a, y = (np.array([i]) for i in (user[self.indices], action, outcome))
        self.policy.partial_fit(x, a, y)
        self.observations.loc[len(self.observations)] = np.append(user, [action, outcome])


    def final_analysis(self) -> None:
        """
        After all the data has been obtained, do a final analysis. This can consist of a number of things:
        1. Recommending a specific fixed treatment policy
        2. Suggesting looking at specific genes more closely
        3. Showing whether or not the new treatment might be better than the old, and by how much.
        4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
        """
        cured = self.observations["outcome"] == 1
        print("The adaptive policy had a ", cured.sum()/len(self.observations), "curing rate")


        cured_explor = cured.iloc[:self.exploit_after_n]
        explor_obs = self.observations.iloc[:self.exploit_after_n]
        explor_total = Counter(explor_obs["action"])
        explor_cured = Counter(explor_obs[cured_explor]["action"])
        explor_curing_rates = {k:explor_cured[k]/explor_total[k] for k in explor_cured}
        print(explor_curing_rates)
        explor_df = pd.DataFrame(explor_curing_rates, ["actions taken"])
        explor_df.plot.bar(rot=0, title="Curing treatments given in exploration phase")

        if self.exploit_after_n < len(self.observations):
            cured_exploit = cured.iloc[self.exploit_after_n:]
            exploit_obs = self.observations.iloc[self.exploit_after_n:]
            exploit_total = Counter(exploit_obs["action"])
            exploit_cured = Counter(exploit_obs[cured_exploit]["action"])
            exploit_curing_rates = {k:exploit_cured[k]/exploit_total[k] for k in exploit_cured}
            exploit_df = pd.DataFrame(exploit_curing_rates, ["actions taken"])
            exploit_df.plot.bar(rot=0, title="Curing rates given in exploitation phase")
            print(exploit_total)
            print(exploit_curing_rates)
        else:
            print(explor_total)

        best_action, best_curing_rate = max(exploit_curing_rates.items(), key=lambda t: t[1])
        print("1: recommending a fixed treatment policy for action", max(exploit_curing_rates, key=lambda k: exploit_curing_rates[k]))

        if self.n_features is not None and self.n_actions > 2:
            print("2: look more into genes:", *(f"gen_{i-2}" for i, _ in explor_cured.most_common(6) if i > 2))

        if self.n_actions > 2:
            print("3: Curing rate for old treatment:", exploit_curing_rates.get(1, "NOT CHOSEN"), "curing rate for new treatment: ", exploit_curing_rates.get(2, "NOT CHOSEN"))

            print("4: Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment")

            best_gene, gene_curing_rate = max(explor_curing_rates.items(), key=lambda t: t[1] * t[1] > 2)

            p = best_curing_rate / gene_curing_rate
            print("We estimate that fixed treatment is ", abs(p-1), "times", "better" if p > 0 else "worse", " than the best gene targeting treatment with a curing rate of", )
            print("Best curing rate: ", best_curing_rate)
            print("Best gene curing rate: ", gene_curing_rate)
