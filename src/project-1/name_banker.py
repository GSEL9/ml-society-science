import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


class NameBanker:
    """DOCS
    """

    # actions = (0, 1)
    # classes = (1, 2)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Docstring for fit"""

        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=0,
            # TODO: find best parameters
            # max_depth=self.best_max_depth,
            # max_features=self.best_max_features,
            class_weight="balanced"
        )
        self.classifier.fit(X,y)

    def set_interest_rate(self, rate: float) -> None:

        self.rate = rate

    def predict_proba(self, x: pd.Series) -> float:
        """Returns the probability that a person will return the loan."""

        if not hasattr(self, "classifier"):
            raise ValueError("This NameBanker instance is not fitted yet. Call 'fit' "
                             "with appropriate arguments before using this method.")

        return self.classifier.predict_proba([x.to_numpy()])[0][1]

    def expected_utility(self, x: pd.Series, action: int) -> float:
        """Calculate the expected utility of a particular action for a given individual.

        Args:
            x: Features describing a selected individual
            action: whether or not to give loan.

        Returns:
            probability:
                real number between 0 and 1 denoting probability of returning loan
                given the features
        """

        if action:

            # Probability of being credit worthy.
            pi = self.predict_proba(x)

            return x["amount"] * ((1 + self.rate) ** x["duration"] - 1) * pi - x["amount"] * (1 - pi)

        return 0.0

    def get_best_action(self, x: pd.Series) -> int:
        """Returns the action maximising expected utility.

        Args:
            x: Feature "vector" describing a selected individual
        Returns:
            action: 0 or 1 regarding wether or not to give loan
        """
        return int(self.expected_utility(x, 1) > 0)


if __name__ == "__main__":

    target = ['repaid']
    features = ['checking account balance', 'duration', 'credit history',
                'purpose', 'amount', 'savings', 'employment', 'installment',
                'marital status', 'other debtors', 'residence time',
                'property', 'age', 'other installments', 'housing', 'credits',
                'job', 'persons', 'phone', 'foreign']

    df = pd.read_csv('../../data/credit/german.data', sep=' ', names=features + target)

    decision_maker = NameBanker()
    decision_maker.set_interest_rate(0.05)
    utility = decision_maker.expected_utility(df, 1)
    print(utility)
