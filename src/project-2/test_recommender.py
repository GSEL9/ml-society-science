import argparse
import numpy as np
import pandas as pd

import data_generation

from recommenders import policies


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("policy", type=str)
    p.add_argument("n_tests", nargs="?", type=int, default=1000)
    p.add_argument("-s","--seed", type=int, default=None)
    return p.parse_args()


def default_reward_function(action, outcome):
    return -0.1 * (action != 0) + outcome


def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        # print("x: ", x, "a: ", a, "y:", y, "r:", r)
    return u


def main(args):
    n_tests = args.n_tests

    features = pd.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
    actions = pd.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
    outcome = pd.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values
    observations = features[:, :128]
    labels = features[:, 128] + features[:,129] * 2

    policy_factory = policies[args.policy]

    descriptions = ["Single treatment", "Additional treatment"]

    max_reward, opt_policy = 0, 0
    for i, description in enumerate(descriptions):

        print(description)

        # print("Setting up simulator")
        generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat", seed=args.seed)
        # print("Setting up policy")
        n_actions = generator.get_n_actions()
        n_outcomes = generator.get_n_outcomes()
        print("n actions:", n_actions, "n_outcomes", n_outcomes)
        policy = policy_factory(n_actions, n_outcomes)
        ## Fit the policy on historical data first
        print("Fitting historical data to the policy")
        policy.fit_treatment_outcome(features, actions, outcome)
        ## Run an online test with a small number of actions
        print("Running an online test")
        result = test_policy(generator, policy, default_reward_function, n_tests)
        print("Total reward:", result)
        print("*** Final analysis of recommender ***")
        policy.final_analysis()

        if max_reward < result:
            max_reward = result
            opt_policy = description

    print("Recommended policy:", opt_policy)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
