import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class NameBanker:
    """DOCS
    """

    # actions = (0, 1)
    # classes = (1, 2)

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fits a model for calculating the probability of credit- worthiness

        Args:
            X: Feature set of individuals
            y: Target labels against the individuals

        Returns:
            None

        """

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        #Finding optimal paramters
        param_grid = [{
            'bootstrap' : [True],
            'max_features' : list(range(10,20,1)),
            'max_depth' : list(range(10,100,10)),
            'n_estimators' : list(range(25,150,25))
        }]

        grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, cv = 5)
        grid_search.fit(X_scaled,y)

        self.classifier = RandomForestClassifier(**grid_search.best_params_)

        #self.classifier = RandomForestClassifier(
            #n_estimators=100,
            #random_state=42,
            # TODO: find best parameters
            # max_depth=self.best_max_depth,
            # max_features=self.best_max_features,
            #class_weight="balanced"
        #)
        

        self.classifier.fit(X,y)

    def set_interest_rate(self, rate: float) -> None:

        self.rate = rate

    def predict_proba(self, x: pd.Series) -> float:
        """Returns the probability that a person will return the loan.

        Args:
            x: Features describing a selected individual

        Returns:
            probability:
                real number between 0 and 1 denoting probability of the individual to be credit-worthy

        """

        if not hasattr(self, "classifier"):
            raise ValueError("This NameBanker instance is not fitted yet. Call 'fit' "
                             "with appropriate arguments before using this method.")

        x_reshaped = np.reshape(x.to_numpy(), (1,-1))
        
        x_scaled = self.scaler.transform(x_reshaped)
        
        return self.classifier.predict_proba(x_scaled)[0][1]

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
