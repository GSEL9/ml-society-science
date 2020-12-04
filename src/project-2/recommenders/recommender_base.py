from abc import abstractmethod, ABC

import sklearn
import pandas as pd 


def feature_counter():

    features = ["sex", "smoker"] + [f"gen_{i}" for i in range(128)] + ["action", "outcome"]

    return pd.DataFrame({c: [] for c in features})


class Recommender(ABC):
    """
    Abstract base class for recommender
    """
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

        self.observations = feature_counter()

    def _default_reward(self, action, outcome):
        "By default, the reward is just equal to the outcome, as the actions play no role."
        return outcome

    def set_reward(self, reward):
        "Set the reward function r(a, y)"
        self.reward = reward


    @abstractmethod
    def fit_treatment_outcome(self, data, actions, outcome):
        """
        Fit a model from patient data, actions and their effects
        Here we assume that the outcome is a direct function of data and actions
        This model can then be used in estimate_utility(), predict_proba() and recommend()
        """
        pass


    @abstractmethod
    def recommend(self, user_data):
        """
        Return recommendations for a specific user datum
        This should be an integer in range(self.n_actions)
        """
        pass


    @abstractmethod
    def observe(self, user, action, outcome):
        """
        Observe the effect of an action. This is an opportunity for you
        to refit your models, to take the new information into account.
        """
        pass


    @abstractmethod
    def final_analysis(self):
        """
        After all the data has been obtained, do a final analysis. This can consist of a number of things:
        1. Recommending a specific fixed treatment policy
        2. Suggesting looking at specific genes more closely
        3. Showing whether or not the new treatment might be better than the old, and by how much.
        4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
        """
        pass


    def estimate_utility(self, data, actions, outcome, policy=None):
        """
        Estimate the utility of a specific policy from historical data (data, actions, outcome),
        where utility is the expected reward of the policy.

        If policy is not given, simply use the average reward of the observed actions and outcomes.

        If a policy is given, then you can either use importance
        sampling, or use the model you have fitted from historical data
        to get an estimate of the utility.

        The policy should be a recommender that implements get_action_probability()
        """
        pass


    def predict_proba(self, data, treatment):
        """
        Return a distribution of effects for a given person's data and a specific treatment.
        This should be an numpy.array of length self.n_outcomes
        """
        pass


    def get_action_probabilities(self, user_data):
        """
        # Return a distribution of recommendations for a specific user datum
        # This should a numpy array of size equal to self.n_actions, summing up to 1
        """
        pass


    def fit_data(self, data):
        """
        Fit a model from patient data.

        This will generally speaking be an
        unsupervised model. Anything from a Gaussian mixture model to a
        neural network is a valid choice.  However, you can give special
        meaning to different parts of the data, and use a supervised
        model instead.
        """
        pass
