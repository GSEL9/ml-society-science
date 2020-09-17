import pandas as pd
from collections import Counter
from itertools import chain

# for backwards portability, define prod if not in math module
try:
    from math import prod
except ImportError:
    import functools, operator
    def prod(iterable):
        return functools.reduce(operator.mul, iterable, 1)

class NameBanker:
    actions = (0, 1)
    classes = (1, 2)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Docstring for fit"""
        self.features = X
        self.targets = y
        self.N = len(X)

        self.feature_frequencies = {col: Counter(X[col]) for col in X.columns}
        print(self.feature_frequencies)
        self.conditionals = {c: {col: Counter(X[y == c][col]) for col in X.columns} for c in self.classes}

        self.class_frequencies = {c : len(y[y==c]) for c in self.classes}

        # for counts in chain(self.feature_frequencies.values(), self.conditionals.values()):
        for counts in self.feature_frequencies.values():
            counts["sum"] = sum(counts.values())

    def set_interest_rate(self, rate: int) -> None:
        self.rate = rate

    # predicting credit worthiness as input to your policy
    def predict_proba(self, x: pd.DataFrame) -> float:
        """The probability that a person will return the loan."""

        # p(c | x) = (p(x) * p(x | c))/p(c)

        # prod()
        # class space:
        # 1 == return_loan
        # 2 == not_return_loan
        p_c = self.class_frequencies[2]/self.N

        # alpha to apply smoothing
        alpha = 1.4e-9

        p_x = prod((self.feature_frequencies[col].get(x[col], 0) + alpha)
                    / (self.feature_frequencies[col]["sum"] + alpha)
                    for col in self.features.columns)

        p_x_given_c = prod((self.conditionals[1][col].get(x[col], 0) + alpha)
                            / (self.class_frequencies[1] + alpha)
                            for col in self.features.columns)

        return (p_x * p_x_given_c)/p_c

    def expected_utility(self, x: pd.DataFrame, action: int) -> float:
        """Calculate the expected utility of a particular action for a given individual.

        Args:
            x: Features describing a selected individual
            action: whether or not to give loan.
        """

        if "features" not in self.__dict__:
            raise ValueError("This NameBanker instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        n = x["duration"]
        m = x["credits"]

        if action:
            p = self.predict_proba(x)
            return p * m * ((1+self.rate)**n - 1)
        else:
            p = 1 - self.predict_proba(x)
            return p * m

    def get_best_action(self, x: pd.DataFrame) -> int:
        """Return the action maximising expected utility.

        Args:
            x: Features describing a selected individual
        Returns:
            action: 0 or 1 regarding wether or not to give loan
        """

        # Referring to the dataset description in 'german.doc' :
        #
        # It is worse to class a customer as good when they are bad (5),
        # than it is to class a customer as bad when they are good (1).

        # TODO: weight the costs so that it selects the best action with respect
        # to the uneven cost balance
        return max(self.actions, key=lambda a: self.expected_utility(x, a))
