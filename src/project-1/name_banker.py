import numpy as np 


class NameBanker:

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
