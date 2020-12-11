

# class AdaptiveRecommender(Recommender):
#     """
#     An adaptive recommender for active treatment. Based on context bandit
#     """
#
#     def __init__(self, n_actions, n_outcomes, seed=None):
#         "Initialize alpha and beta as model parameters"
#         super().__init__(n_actions, n_outcomes)
#
#         self.rng = np.random.RandomState(seed or 0)
#
#
#
#
#     def fit_treatment_outcome(self, data: np.ndarray,
#                                     actions: np.ndarray,
#                                     outcome: np.ndarray,
#                                     random_state:int=0):
#         pass
#
#     def observe(self, user, action, outcome):
#         """
#         x: Context vector
#         a: action
#         y: outcome
#         """
#         x, a, y = user, action, outcome
#
#         self.alphas[a][x] += y
#         self.betas[a][x] += (1 - y)
#
#     def recommend(self, user_data):
#         """
#         x: Context vector
#         actions_t: local action space
#         """
#         if not hasattr(self, "alphas"):
#             self.init_parameters(len(user_data))
#
#         p_t_a = self.rng.beta(self.alphas, self.betas)
#         print(p_t_a)
#         print(p_t_a.shape)
#         a_t = np.argmax(p_t_a)
#         print(a_t)
#         return a_t
#
#     def init_parameters(self, n_features):
#
#         self.alphas = self.rng.uniform(size=(self.n_actions, n_features))
#         self.betas = self.rng.uniform(size=(self.n_actions, n_features))
#
#
#     def final_analysis(self):
#         """Shows which genetic features to look into and a success rate for the treatments"""
