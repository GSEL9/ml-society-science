import numpy as np
import pandas as pd
from group4_banker import Group4Banker
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from test_lending import setup_data
from functools import partial

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("optinal package seaborn not found: proceeding with default theme")

single_male = "marital status_3"
single_female = "marital status_5"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n-tests", type=int, default=3)
    parser.add_argument("-r", "--interest-rate", type=float, default=0.05)
    parser.add_argument("-s", "--seed", type=int, default=42)

    return parser.parse_args()

def get_trained_model(interest_rate):
    X_train, feature_data = setup_data("../../data/credit/D_train.csv")
    decision_maker = Group4Banker()
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train[feature_data["encoded_features"]],
                       X_train[feature_data["target"]])
    return decision_maker

def get_encoded_features():
    return setup_data("../../data/credit/D_train.csv")[1]["encoded_features"]


def measure_probability_difference(args):
    np.random.seed(args.seed)
    X_train, feature_data = setup_data("../../data/credit/D_train.csv")
    X_val, *_ = setup_data("../../data/credit/D_valid.csv")

    encoded_features = feature_data["encoded_features"]
    target = feature_data["target"]

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


def create_histogram(args):
    dataset, feature_data = setup_data("../../data/credit/D_train.csv")
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    X = dataset[encoded_features]
    y = dataset[target]
    decision_maker = Group4Banker()
    decision_maker.set_interest_rate(0.05)
    decision_maker.fit(X, y)
    # plt.subplots(1, 2)

    # dataset, feature_data = setup_data("../../data/credit/D_train.csv")
    # ax_1 = generate_barchart(decision_maker, dataset, feature_data, single_male)
    # plt.show()
    # ax_2 = generate_barchart(decision_maker, dataset, feature_data, single_female)
    # plt.show()

    # genereate_histogram_outcome(decision_maker, dataset, feature_data, single_male)
    # plt.show()
    generate_histogram_utility(decision_maker, dataset, feature_data, single_female)
    plt.show()


def generate_barchart(decision_maker, dataset, feature_data, feature, target_value=1, plot_title=None):
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    dataset = dataset[dataset[feature] == target_value]

    X = dataset[encoded_features]
    y = dataset[target]
    probs = decision_maker.classifier.predict_proba(X).T[0]

    ax = plt.axis()
    plt.bar(X[y == 1]["amount"], probs[y == 1], width=2000, color="blue", label="returned loan")
    plt.bar(X[y == 2]["amount"], probs[y == 2], width=2000, color="orange", label="did not return loan")
    plt.xlabel("loan amount")
    plt.ylabel("predicted probability")
    plt.title(f"predicted probability considering loan for {feature}")
    return ax


def generate_histogram_outcome(decision_maker, dataset, feature_data, feature, target_value=1, ax=None):
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    dataset = dataset[dataset[feature] == target_value]

    X = dataset[encoded_features]
    y = dataset[target]
    if ax:
        ax.hist(X[y == 1]["amount"], 20, label="returned loan", alpha=.5, color="blue")
        ax.hist(X[y == 2]["amount"], 20, label="did not return loan", alpha=.8, color="orange")

        ax.set_xlabel("amount of loan")
        ax.set_ylabel("amount of applicants")
        ax.legend()
    else:
        plt.hist(X[y == 1]["amount"], 20, label="returned loan", alpha=.5, color="blue")
        plt.hist(X[y == 2]["amount"], 20, label="did not return loan", alpha=.8, color="orange")
        plt.set_xlabel("amount of loan")
        plt.set_ylabel("amount of applicants")
        plt.legend()


def generate_histogram_action(decision_maker, dataset, feature_data, feature, target_value=1, ax=None):
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    dataset = dataset[dataset[feature] == target_value]

    X = dataset[encoded_features]

    y_hat = X.apply(decision_maker.get_best_action, axis=1)

    if ax:
        ax.hist(X[y_hat == 1]["amount"], 20, label="granted loan", alpha=.5, color="blue")
        ax.hist(X[y_hat == 0]["amount"], 20, label="refused loan", alpha=.8, color="orange")

        ax.set_xlabel("amount of loan")
        ax.set_ylabel("amount of applicants")
        ax.legend()
    else:
        plt.hist(X[y_hat == 1]["amount"], 20, label="granted loan", alpha=.5, color="blue")
        plt.hist(X[y_hat == 0]["amount"], 20, label="refused loan", alpha=.8, color="orange")

        plt.xlabel("amount of loan")
        plt.ylabel("amount of applicants")
        plt.legend()


def generate_histogram_utility(decision_maker, dataset, feature_data, feature, target_value=1, ax=None):
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    dataset = dataset[dataset[feature] == target_value]

    X = dataset[encoded_features]
    U = X.apply(partial(decision_maker.expected_utility, action=1), axis=1)
    U = 1/(1+np.exp(-U.to_numpy()))
    y = dataset[target]


    if ax:
        ax.hist(U[y==1], 20, label="granted loan", alpha=.5, color="blue")
        ax.hist(U[y==2], 20, label="refused loan", alpha=.8, color="orange")

        ax.set_xlabel("expected_utility")
        ax.set_ylabel("amount of applicants")
        ax.legend()
    else:
        plt.hist(U[y==1], 20, label="granted loan", alpha=.5, color="blue")
        plt.hist(U[y==2], 20, label="refused loan", alpha=.8, color="orange")

        plt.xlabel("amount of loan")
        plt.ylabel("amount of applicants")
        plt.legend()

def main():
    args = parse_args()
    measure_probability_difference(args)

    # create_histogram(args)

if __name__ == "__main__":
    main()
