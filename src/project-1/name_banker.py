import numpy as np 
from sklearn.ensemble import RandomForestClassifier

class NameBanker:

	#Fit a model for calculating the probability of credit-worthiness
	def fit(self, x, y):
		#TODO: Preprocessing
		self.classifier = RandomForestClassifier(
			n_estimators=100,
			random_state=0,
			max_depth=self.best_max_depth,
			max_features=self.best_max_features,
			class_weight="balanced"
			)
		self.classifier.fit(x,y)

	#Return the probability that a person will return the loan
	def predict_proba(self, x):
		#TODO: some reshaping or preprocessing
		prediction = self.classifier.predict_proba(x)[0][1]
		return prediction



	def set_interest_rate(self, rate):

		self.rate = rate

	# TODO:
	# TEMP: Dummy config 
	def get_proba(self):

		return 0.4

	def expected_utility(self, x, action):

		# Expected utility given some action.
		utility = 0
		for i in range(len(x)):

			# No update unless granting loan.
			if action == 1:

				# Probability of being credit worthy.
				Pi = self.get_proba()

				n = x["duration"].iloc[i]
				m = x["amount"].iloc[i]

				utility += m * ((1 + self.rate) ** n - 1) * Pi - (1 - Pi) * m

		return utility

	



if __name__ == "__main__":

	import pandas

	target = ['repaid']
	features = ['checking account balance', 'duration', 'credit history',
	            'purpose', 'amount', 'savings', 'employment', 'installment',
	            'marital status', 'other debtors', 'residence time',
	            'property', 'age', 'other installments', 'housing', 'credits',
	            'job', 'persons', 'phone', 'foreign']

	df = pandas.read_csv('../../data/credit/german.data', sep=' ', names=features + target)

	decision_maker = NameBanker()
	decision_maker.set_interest_rate(0.05)
	utility = decision_maker.expected_utility(df, 1)
	print(utility)
