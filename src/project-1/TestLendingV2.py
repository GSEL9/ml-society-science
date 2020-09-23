import pandas
import matplotlib.pyplot as plt
from random_banker import RandomBanker
from name_banker import NameBanker
from markus_banker import MarkusBanker
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm


def setup_data(data_path):
    ## Set up for dataset
    features = ['checking account balance', 'duration', 'credit history',
                'purpose', 'amount', 'savings', 'employment', 'installment',
                'marital status', 'other debtors', 'residence time',
                'property', 'age', 'other installments', 'housing', 'credits',
                'job', 'persons', 'phone', 'foreign']

    target = 'repaid'
    # df = pandas.read_csv('../../data/credit/german.data', sep=' ',
    #                     names=features+[target])
    df = pandas.read_csv(data_path, sep=' ',
                        names=features+[target])

    numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']
    quantitative_features = list(filter(lambda x: x not in numerical_features, features))
    X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
    encoded_features = list(filter(lambda x: x != target, X.columns))

    return X, encoded_features, target


## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    n_test_examples = len(X_test)
    utility = 0

    ## Example test function - this is only an unbiased test if the data has not been seen in training
    total_amount = 0
    total_utility = 0
    decision_maker.set_interest_rate(interest_rate)
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        good_loan = y_test.iloc[t] # assume the labels are correct
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        if (action==1):
            if (good_loan != 1):
                utility -= amount
            else:
                utility += amount*(pow(1 + interest_rate, duration) - 1)
        total_utility += utility
        total_amount += amount
    return utility, total_utility/total_amount


def parse_args():
    ap = ArgumentParser()
    ap.add_argument("data_path", nargs="?", default='../../data/credit/D_valid.csv')
    ap.add_argument("--n-tests", type=int, default=10)
    ap.add_argument("-r", "--interest-rate", type=float, default=0.017)
    # random seed
    # ap.add_argument()
    
    return ap.parse_args()

def main():
    np.random.seed(42)
    args = parse_args()

    X, encoded_features, target = setup_data(args.data_path)

    ### Setup model
    interest_rate = args.interest_rate
    ### Do a number of preliminary tests by splitting the data in parts
    n_tests = args.n_tests

    print(f"r={interest_rate}")
    print(f"n_tests={n_tests}")

    for decision_maker in RandomBanker(), NameBanker():
        decision_maker.set_interest_rate(interest_rate)
        utility = 0
        investment_return = 0
        for i in tqdm(range(n_tests)):
            X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
            decision_maker.set_interest_rate(interest_rate)
            decision_maker.fit(X_train, y_train)
            Ui, Ri = test_decision_maker(X_test, y_test, interest_rate, decision_maker)
            utility += Ui
            investment_return += Ri

        print("Testing on class:", type(decision_maker).__name__)
        print("\tAverage utility:", utility / n_tests)
        print("\tAverage return on investment:", investment_return / n_tests)


if __name__ == "__main__":
    main()
