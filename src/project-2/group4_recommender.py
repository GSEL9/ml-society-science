from tensorflow import keras
from sklearn import linear_model
import numpy as np

from estimate_utility import expected_utility
from typing import Iterable


def utility(y, a):
    return -0.1 * a + y


class Group4Recommender:
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(1, input_shape=(130,), activation="sigmoid", dtype="float32"))
        self.model.compile(optimizer="sgd", loss=utility)


    def _default_reward(self, action, outcome):
        return outcome


    def set_reward(self, reward):
        self.reward = reward


    def fit_data(self, data):
        print("Preprocessing data")
        return None


    def fit_treatment_outcome(self, data, actions, outcome):
        # print("Fitting treatment outcomes")
        self.model.fit(data, outcome)


    def estimate_utility(self, data: np.ndarray, actions: Iterable[bool], outcome: Iterable[bool], policy=None):
        assert np.ndim(actions) == 1
        assert np.ndim(outcome) == 1

        if policy is not None:

            # TEMP: Fixed model params.
            policy_actions = take_action(data, actions, outcome,
                                         penalty='l1', max_iter=1000)

            return expected_utility(data, policy_actions, outcome, policy, return_ci=True)

        else:

            # Utility is the expected reward
            r = -0.1 * actions + outcome

            return -1.0 * r.std(), r.mean(), r.std()
        
    # def predict_proba(self, data, treatment):
        # return np.zeros(self.n_outcomes)


    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        #print("Recommending")
        return np.ones(self.n_actions) / self.n_actions;

    def recommend(self, user_data):
        # return np.random.choice(self.n_actions, p = self.get_action_probabilities(user_data))
        return self.model(user_data)

    def observe(self, user, action, outcome):
        return None

    def final_analysis(self):
        return None
