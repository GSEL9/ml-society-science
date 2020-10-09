import numpy as np
import pandas as pd
from group4_banker import Group4Banker
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from TestLendingV2 import setup_data

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n-tests", type=int, default=3)
    parser.add_argument("-r", "--interest-rate", type=float, default=0.05)
    parser.add_argument("-s", "--seed", type=int, default=42)

    return parser.parse_args()

def get_trained_model(interest_rate):
    X_train, encoded_features, target = setup_data("../../data/credit/D_train.csv")
    decision_maker = Group4Banker()
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train[encoded_features], X_train[target])
    return decision_maker

def get_encoded_features():
    return setup_data("../../data/credit/D_train.csv")[1]

def main(args):
    np.random.seed(args.seed)

    X_train, encoded_features, target = setup_data("../../data/credit/D_train.csv")
    X_val, *_ = setup_data("../../data/credit/D_valid.csv")

    single_male = "marital status_3"
    single_female = "marital status_5"

    decision_maker = Group4Banker()
    decision_maker.set_interest_rate(args.interest_rate)
    decision_maker.fit(X_train[encoded_features], X_train[target])

    df_single_male = (X_train[single_male] == 1)
    single_male_and_return = (X_train[single_male] == 1) & (X_train[target] == 1)

    df_single_female = (X_train[single_female] == 1)
    single_female_and_return = (X_train[single_female] == 1) & (X_train[target] == 1)

    print("Proportion of returns for single males in the train set:\n", single_male_and_return.sum()/df_single_male.sum())
    print("Proportion of returns for single females in the train set:\n", single_female_and_return.sum()/df_single_female.sum(), "\n")


    samples = X_val.sample(n=args.n_tests)

    for i, row in samples.iterrows():
        for i in range(1, 6):
            row["martial status_"+str(i)] = 0

        row[single_male] = 1

        proba_on_m = decision_maker.predict_proba(row[encoded_features])
        utility_m = decision_maker.expected_utility(row[encoded_features], 1)

        row[single_male] = 0
        row[single_female] = 1

        proba_on_f = decision_maker.predict_proba(row[encoded_features])
        utility_f = decision_maker.expected_utility(row[encoded_features], 1)
        row[single_female] = 0

        print("Estimated probability for single male:", proba_on_m,
            "\nEstimated probabiltiy for single female:", proba_on_f,
            "\nAbsolute difference", abs(proba_on_m - proba_on_f), "\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)
