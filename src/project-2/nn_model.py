import numpy as np 
import tensorflow as tf 

from tensorflow import keras


DTYPE = tf.float32


class Linear(keras.layers.Layer):

	def __init__(self, units=32):

		super(Linear, self).__init__()

		self.units = units

	def build(self, input_shape):

		self.w = self.add_weight(shape=(input_shape[-1], self.units),
			initializer="random_normal",
			trainable=True,
			dtype=DTYPE
		)

		self.b = self.add_weight(shape=(self.units,), dtype=DTYPE,
								 initializer="random_normal", trainable=True)

	def call(self, data):
		return tf.matmul(data, self.w) + self.b


class ANN(tf.keras.Model):

	def __init__(self):

		super(ANN, self).__init__()

		self.linear_1 = Linear(32)
		self.linear_2 = Linear(32)
		self.linear_3 = Linear(1)

	def call(self, data):

		data = tf.cast(data, dtype=DTYPE)

		x = self.linear_1(data)
		x = tf.nn.relu(x)
		x = self.linear_2(x)
		x = tf.nn.relu(x)

		return tf.math.sigmoid(self.linear_3(x))


def loss(model, data, outcome, training=False):

	p0_hat = model(data, training=training)

	return -1.0 * tf.reduce_sum(outcome * p0_hat + (outcome - 0.1) * (1 - p0_hat))


def grad(model, data, outcome):
	
	with tf.GradientTape() as tape:
		loss_value = loss(model, data, outcome, training=True)

	return loss_value, tape.gradient(loss_value, model.trainable_variables)


# QUESTION: Save loss?
def train_ann_model(data: np.ndarray, actions: np.ndarray, outcome: np.ndarray,
	  			    n_epochs: int=100, learning_rate: float=0.001, batch_size=0):

	dset_data = tf.data.Dataset.from_tensor_slices(data)
	dset_outcome = tf.data.Dataset.from_tensor_slices(outcome)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	
	model = ANN()
	for _ in range(n_epochs):

		if batch_size > 0:

			data_batches = dset_data.batch(batch_size)
			outcome_batches = dset_outcome.batch(batch_size)

			for data_i, outcome_i in zip(data_batches, outcome_batches):

				loss_value, grads = grad(model, data_i, outcome_i.numpy())

				optimizer.apply_gradients(zip(grads, model.trainable_variables))

		else:

			loss_value, grads = grad(model, data, outcome)

			optimizer.apply_gradients(zip(grads, model.trainable_variables))

	return model 


if __name__ == "__main__":

	import numpy as np 
	import pandas as pd 

	from estimate_utility import expected_utility

	data = pd.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
	actions = pd.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
	outcome = pd.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values

	# NB: Should kill redundant dimension.
	outcome = np.squeeze(outcome)

