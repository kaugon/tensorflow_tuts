#####################################
## Copyright 2017 @ Kau Gon
####################################

import tensorflow as tf
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

DATA_FILE = './data/fire_theft_csv.csv'
TEST_SIZE = 10
EPOCH_SIZE = 100

def read_data_file(data_file):
	print "Reading Data: %s" % data_file
	data = []
	with open(data_file, mode='r') as fp:
		csvr = csv.DictReader(fp)
		data = np.asarray([[row['X'], row['Y']] for row in csvr], dtype=np.float32)
	#print data
	print data.shape, data.dtype
	return data

def huber_loss(y, y_predict, delta=1.0):
	residual = tf.abs(y - y_predict)
	condition = tf.less(residual, delta)
	delta_loss = 0.5 * tf.square(residual)
	outlier_loss = (delta * residual) - (0.5 * tf.square(delta))
	h_loss = tf.where(condition, delta_loss, outlier_loss)
	return h_loss

def linear_regression(data):
	# x_input, y_expected
	x = tf.placeholder(tf.float32, name="x")
	y = tf.placeholder(tf.float32, name="y")

	# parameters to tune	
	weights = tf.Variable(0, dtype=tf.float32, name="weights")
	biases = tf.Variable(0, dtype=tf.float32, name="biases")

	# y = (x * W) + b
	y_predict = x * weights + biases

	# square error is the loss
	loss_square = tf.square(y - y_predict, name="loss_square")
	loss = huber_loss(y, y_predict)

	# minimize loss 
	# fradient descent for back propogation 
	# dependent tfVariables are updated automatically 
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001, name="GD_Optimizer").minimize(loss) 

	# read data file 
	# data = read_data_file(DATA_FILE)
	data_size = len(data)
	test_size = TEST_SIZE
	train_size = data_size # - test_size

	# tf session execution
	with tf.Session() as sess:
		# global variables init
		sess.run(tf.global_variables_initializer())

		# log everything
		writer = tf.summary.FileWriter("./linear_regress", sess.graph)

		print "Training model.."
		# epochs
		for epoch in range(EPOCH_SIZE):
			total_loss = 0
			for dx, dy in data[:train_size]:
				co, cl = sess.run([optimizer, loss], feed_dict={x:dx, y:dy})
				total_loss += cl
			print "Epoch: %s/%s, Loss: %s" % (epoch, EPOCH_SIZE, total_loss/train_size) 
 
		w_val, b_val = sess.run([weights, biases])

	print "w: %s, b: %s" % (w_val, b_val)
	writer.close()

	return w_val, b_val

def plot_vals(data, w, b):
	print "Plotting results.."
	X, Y = data[:,0], data[:,1]
	#print X
	#print Y	
	plt.plot(X, Y, 'bo', label='Real data')
	plt.plot(X, X*w+b, 'r', label="Predicted")
	plt.legend()
	plt.show()

if __name__ == '__main__':
	data = read_data_file(DATA_FILE) 
	w, b = linear_regression(data)
	plot_vals(data, w, b)	
