#####################################
## Copyright 2017 @ Kau Gon
####################################

import os
import numpy as np
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

BATCH_SIZE = 128
# in this case, higher epoch doesnt help much
# i.e. limitations of logistical/linear regression model being single layer
EPOCH_SIZE = 40
RESTORE_MODEL = True 

class LogisticRegressionModel(object):
	def __init__(self, batch_size):
		self._batch_size = batch_size

	def create_placeholders(self):
		# x_input, label_expected
		self._x = tf.placeholder(tf.float32, shape=[self._batch_size, 784], name="image")
		self._y = tf.placeholder(tf.int32, shape=[self._batch_size, 10], name="labels")

	def create_model(self):
		# parameters to tune
		# this is not convolution hence weights shape is [input, output]	
		self._weights = tf.Variable(tf.random_normal(shape=[784, 10],
					stddev=0.01), 
					name="weights")
		self._biases = tf.Variable(tf.zeros(shape=[1,10]),
					name="biases")

		# y = (x * W) + b
		self._y_predict = tf.add(tf.matmul(self._x, self._weights), self._biases)

	def create_loss(self):
		# softmax and loss combined in single tf function
		self._y_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._y_predict,
					labels=self._y, name="y_softmax")
		# y_entropy size [batch_size, 1]
		self._loss = tf.reduce_mean(self._y_entropy)

	def create_optimizer(self):
		# minimize loss 
		# gradient descent for back propogation 
		# dependent tfVariables are updated automatically 

		# Adam Optimizer does better job than GradientDescent
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss) 
		adamop = tf.train.AdamOptimizer(learning_rate=0.001)
		self._optimizer = adamop.minimize(self._loss) 

	def create_summaries(self):
		with tf.name_scope("summaries"):
			tf.summary.scalar("loss", self._loss)
			tf.summary.histogram("loss_hist", self._loss)
			self._summary_op = tf.summary.merge_all()

	def build_model(self):
		self.create_placeholders()
		self.create_model()
		self.create_loss()
		self.create_optimizer()
		self.create_summaries()
		
	def load_data(self):
		# mnist data
		self._data = input_data.read_data_sets('/data/mnist', one_hot=True)	
		self._train_batches = int(self._data.train.num_examples/self._batch_size)

	def train_model_per_epoch(self, sess):
		# mnist train data set
		total_loss = 0
		for i_batch in range(self._train_batches):
			batch_x, batch_y = self._data.train.next_batch(self._batch_size)
			_, cl,summary = sess.run([self._optimizer, self._loss, self._summary_op], 
					feed_dict={self._x:batch_x, self._y:batch_y})
			total_loss += cl
		return total_loss

	def train_model(self, epoch_size):
		# saver 
		saver = tf.train.Saver()

		with tf.Session() as sess:
			# global variables init
			sess.run(tf.global_variables_initializer())
			
			# restore previous session
			ckpt = tf.train.get_checkpoint_state(os.path.dirname('./logs/logit_regress/checkpoints/checkpoint'))
			if RESTORE_MODEL and ckpt and ckpt.model_checkpoint_path:
				print "Restoring checkpoint %s" % ckpt.model_checkpoint_path
				saver.restore(sess, ckpt.model_checkpoint_path)

			# Train model
			print "Training model.. with %s images" % self._data.train.num_examples 
			start_time = time.time()

			# log everything
			logdir = "%s/run_%s" % ("./logs/logit_regress/tbevents/", long(start_time))
			print "Tensorboard run: %s" % logdir
			writer = tf.summary.FileWriter(logdir, sess.graph)

			for epoch in range(1, epoch_size+1):
				# mnist train data set
				total_loss = 0
				for i_batch in range(self._train_batches):
					batch_x, batch_y = self._data.train.next_batch(self._batch_size)
					_, cl,summary = sess.run([self._optimizer, 
								self._loss, 
								self._summary_op], 
							feed_dict={self._x:batch_x, self._y:batch_y})
					total_loss += cl
				writer.add_summary(summary, epoch)

				print "Epoch: %s/%s, Loss: %s" % (epoch, epoch_size, 
						total_loss/self._train_batches)
				# checkpoint training model data (not actual model)
				if (epoch%10) == 0:
					print "Saving checkpoint.."
					saver.save(sess, './logs/logit_regress/checkpoints/model', epoch)
					self.validate_model(sess)
			print "Done. Traing time: %s seconds" % (time.time() - start_time)
		writer.close()

	def validate_model(self, sess):
		# Testing model

		# softmax on y_predict outputs
		y_predict_labels = tf.nn.softmax(self._y_predict)
		# correct prediction
		c_predicts = tf.equal(tf.argmax(y_predict_labels, 1), tf.argmax(self._y, 1))
		# boolean to float32
		# add all elements in this batch for current prediction
		c_batch_predicts = tf.reduce_sum(tf.cast(c_predicts, tf.float32))

		# mnist test data set, not the training data set
		print "Testing model..with %s images" % self._data.test.num_examples
		n_batches = int(self._data.test.num_examples/self._batch_size)

		# we are not training anymore hence epochs not required
		# no need to invoke optimizer, becasue we dont need training/bakcprop
		total_predicts = 0
		for i_batch in range(n_batches):
			batch_x, batch_y = self._data.test.next_batch(self._batch_size)
			results, labels = sess.run([c_batch_predicts, y_predict_labels], 
				feed_dict={self._x:batch_x, self._y:batch_y})
			total_predicts += results
		print "Test Accuracy: %s" % (total_predicts/self._data.test.num_examples)
 

if __name__ == '__main__':
	model = LogisticRegressionModel(BATCH_SIZE)
	model.build_model()
	model.load_data()
	model.train_model(EPOCH_SIZE)
	#model.validate_model()
	
