import numpy as np 


class NameBanker:

	def set_interest_rate(self, rate):

		self.rate = rate

	# TODO:
	# TEMP: Dummy config 
	def get_proba(self):

		return 0.4

	def expected_utility(self, x, action):

		# NOTE: No update unless granting loan.
		if action == 1:

			P_credit_worthy = self.get_proba()
			P_not_credit_worthy = 1 - P_credit_worthy

			U_credit_worthy = x["amount"] * ((1 + self.rate) ** x["duration"] - 1)
			U_not_credit_worthy = -1.0 * x["amount"]

			# Summation over all rewards, r.
			return P_credit_worthy * U_credit_worthy + P_not_credit_worthy * U_not_credit_worthy

		return 0


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
