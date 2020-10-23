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

feature_description = {
    "marital status_1":"divorced/separated male",
    "marital status_2":"divorced/separated/married female",
    "marital status_3":"single male",
    "marital status_4":"married/widowed male",
    "marital status_5":"single female"
}

description_feature = {v:k for k, v in feature_description.items()}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n-tests", type=int, default=3)
    parser.add_argument("-r", "--interest-rate", type=float, default=0.05)
    parser.add_argument("-s", "--seed", type=int, default=42)

    return parser.parse_args()


def get_returns_on_feature(feature, data, target, feature_value=1):
    data = data[data[feature] == 1]
    positives, negatives = data[target] == 1, data[target] == 2
    return positives.sum(), negatives.sum()


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

    single_male = description_feature["single male"]
    single_female = description_feature["single female"]

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

    samples = X_val.sample(n=args.n_tests)
    n_tests = args.n_tests

    avg_diff = 0
    max_diff_male = 0
    max_diff_female = 0

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

        diff = proba_on_m - proba_on_f
        if diff > max_diff_male: max_diff_male = diff
        if diff < max_diff_female: max_diff_female = diff
        absdiff = abs(diff)
        if n_tests < 10:
            print("Estimated probability for single male:", proba_on_m,
                  "\nEstimated probabiltiy for single female:", proba_on_f,
                  "\nAbsolute difference", absdiff, "\n")

    if n_tests >= 10:
        print("Average probability difference:", absdiff/n_tests)
        print("max diff benefitting female:", -max_diff_female)
        print("max diff benefitting male:", max_diff_male)

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


def generate_histogram_outcome(decision_maker, dataset, feature_data, feature, target_value=1, ax=None, title=None):
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
        if title:
            ax.set_title(title)
    else:
        plt.hist(X[y == 1]["amount"], 20, label="returned loan", alpha=.5, color="blue")
        plt.hist(X[y == 2]["amount"], 20, label="did not return loan", alpha=.8, color="orange")
        plt.set_xlabel("amount of loan")
        plt.set_ylabel("amount of applicants")
        plt.legend()


def generate_histogram_action(decision_maker, dataset, feature_data, feature, target_value=1, ax=None, title=None, bottom_label="amount of loan"):
    encoded_features, target = feature_data["encoded_features"], feature_data["target"]
    dataset = dataset[dataset[feature] == target_value]

    X = dataset[encoded_features]

    y_hat = X.apply(decision_maker.get_best_action, axis=1)



    if ax:
        ax.hist(X[y_hat == 1]["amount"], 20, label="granted loan", alpha=.5, color="blue")
        ax.hist(X[y_hat == 0]["amount"], 20, label="refused loan", alpha=.8, color="orange")

        ax.set_xlabel(bottom_label)
        ax.legend()
        ax.set_ylabel("amount of applicants")

        if title:
            ax.set_title(title)

    else:
        plt.hist(X[y_hat == 1]["amount"], 20, label="granted loan", alpha=.5, color="blue")
        plt.hist(X[y_hat == 0]["amount"], 20, label="refused loan", alpha=.8, color="orange")

        plt.xlabel(bottom_label)
        plt.ylabel("amount of applicants")
        plt.legend()


def generate_histogram_utility(decision_maker, dataset, feature_data, feature, title=None, target_value=1, ax=None):
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

        if title:
            ax.set_title(title)
    else:
        plt.hist(U[y==1], 20, label="granted loan", alpha=.5, color="blue")
        plt.hist(U[y==2], 20, label="refused loan", alpha=.8, color="orange")

        plt.xlabel("amount of loan")
        plt.ylabel("amount of applicants")
        plt.legend()


def credit_worthiness_barchart(Y1, Y0, labels, label1='returned loan',
                               label2="no returned loan", legend_anchor=0, **subplot_kwargs):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(**subplot_kwargs)
    rects1 = ax.bar(x - width/2, Y1, width, label=label1, alpha=0.8, color="blue")
    rects2 = ax.bar(x + width/2, Y0, width, label=label2, alpha=0.8, color="orange")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Loan applicants')
    ax.set_title('Distribution of returns')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=legend_anchor)


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()


def main():
    args = parse_args()
    measure_probability_difference(args)

    # create_histogram(args)

if __name__ == "__main__":
    main()
