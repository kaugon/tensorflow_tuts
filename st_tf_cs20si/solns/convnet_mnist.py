#####################################
## Copyright 2017 @ Kau Gon
####################################

import os
import numpy as np
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

MODEL_NAME="convnet_mnist"

# mnist classification
N_CLASSES = 10

BATCH_SIZE = 128
EPOCH_SIZE = 100

# FC dropout rate
DROPOUT_RATE=0.75

# Start from scratch or checkpointed model
RESTORE_MODEL = True
CHECKPOINT_STEP = 1
CHECKPOINT_DIR = "./logs/" + MODEL_NAME + "/checkpoint"
TENSORBOARD_DIR = "./logs/" + MODEL_NAME + "/eventlogs"

class ConvnetModel(object):
    def __init__(self, batch_size):
        self._batch_size = batch_size

    def create_placeholders(self):
        # x_input, label_expected
        # ?? Batch size 
        with tf.name_scope('data'):
            self._x = tf.placeholder(tf.float32, shape=[self._batch_size, 784], name="image")
            self._y = tf.placeholder(tf.int32, shape=[self._batch_size, 10], name="labels")

    def create_model(self):
        """
            Laptop cpu intel i5 - DDR 8GB cant handle this network
            Need GPU:
 
            conv1,      i=28x28x1,  k=5x5x1x32,    s=1x1, o=28x28x32 
            pool1,      i=28x28x32, k=2x2,         s=2x2, o=14x14x32
            conv2,      i=14x14x32, k=5x5x32x54,   s=1x1, o=14x14x64 
            pool2,      i=14x14x64, k=2x2,         s=2x2, o=7x7x64
            fc,         i=7x7x64,   k=7x7x64x1024, s=,    o=1024
            fc_dropout,
            softmax_linear, i=1024, k=1024x10,     s=,    o=10
            softmax,

            Small model:
            Laptop CPU needs less layers and less Ni i.e. 32 to 4
            conv1,      i=28x28x1,  k=5x5x1x4,     s=1x1, o=28x28x4 
            pool1,      i=28x28x4,  k=4x4,         s=4x4, o=7x7x4
            softmax_linear, i=196,  k=196x10,      s=,    o=10
            softmax,

            With this 3 layer model, accuracy hits 97% with 30 epoch. 
            Doesnt improve much beyond it.
         """
        with tf.variable_scope('conv1') as scope:
            inputd = tf.reshape(self._x, shape=[self._batch_size, 28, 28, 1])
            #kernel = tf.Variable(tf.random_normal(shape=[5,5,1,32],stddev=0.01), 
            kernel = tf.Variable(tf.random_normal(shape=[5,5,1,4],stddev=0.01), 
                        name="weights")
            biases = tf.Variable(tf.zeros(shape=[4]),
            #biases = tf.Variable(tf.zeros(shape=[32]),
                        name="biases")
            #conv operation is without bias
            conv1b = tf.nn.conv2d(inputd, kernel, strides=[1,1,1,1], padding="SAME")
            conv1 = tf.nn.relu(conv1b+biases, name=scope.name) 

        with tf.variable_scope('pool1') as scope:
            #pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
            pool1 = tf.nn.max_pool(conv1, ksize=[1,4,4,1], strides=[1,4,4,1], padding="SAME")

        """
        with tf.variable_scope('conv2') as scope:
            kernel = tf.Variable(tf.random_normal(shape=[5,5,32,64],stddev=0.01), 
                        name="weights")
            biases = tf.Variable(tf.zeros(shape=[64]),
                        name="biases")
            conv2b = tf.nn.conv2d(pool1, kernel, strides=[1,1,1,1], padding="SAME")
            conv2 = tf.nn.relu(conv2b+biases, name=scope.name) 

        with tf.variable_scope('pool2') as scope:
            pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

        with tf.variable_scope('fc') as scope:
            kernel = tf.Variable(tf.random_normal(shape=[7*7*64, 1024],stddev=0.01), 
                        name="weights")
            biases = tf.Variable(tf.zeros(shape=[1024]),
                        name="biases")
            pool2_reshaped = tf.reshape(pool2, shape=[-1, 7*7*64])
            fcb = tf.matmul(pool2_reshaped, kernel) 
            fc = tf.nn.relu(fcb+biases, name=scope.name)

        with tf.variable_scope('fc_dropout') as scope:
            fc_dropout = tf.nn.dropout(fc, keep_prob=DROPOUT_RATE, name=scope.name)
        """

        with tf.variable_scope('softmax_linear') as scope:
            #inputd = fc_dropout
            inputd = tf.reshape(pool1, shape=[-1, 7*7*4])
            #kernel = tf.Variable(tf.random_normal(shape=[1024, N_CLASSES],stddev=0.01),
            kernel = tf.Variable(tf.random_normal(shape=[7*7*4, N_CLASSES],stddev=0.01), 
                        name="weights")
            biases = tf.Variable(tf.zeros(shape=[N_CLASSES]),
                        name="biases")
            self._logits = tf.matmul(inputd, kernel) + biases

    def create_loss(self):
        # softmax and loss combined in single tf function
        y_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self._logits,
                    labels=self._y, name="y_softmax")
        self._loss = tf.reduce_mean(y_entropy)

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

    def train_model(self, epoch_size):
        # saver 
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # global variables init
            sess.run(tf.global_variables_initializer())
            
            # restore previous session
            ckpt = tf.train.get_checkpoint_state(
                os.path.dirname(CHECKPOINT_DIR+'/checkpoint'))
            if RESTORE_MODEL and ckpt and ckpt.model_checkpoint_path:
                print "Restoring checkpoint %s" % ckpt.model_checkpoint_path
                saver.restore(sess, ckpt.model_checkpoint_path)

            # Train model
            print "Training model.. with %s images" % self._data.train.num_examples 
            start_time = time.time()

            # log everything
            logdir = "%s/run_%s" % (TENSORBOARD_DIR, long(start_time))
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
                if (epoch%CHECKPOINT_STEP) == 0:
                    print "Saving checkpoint.."
                    saver.save(sess, CHECKPOINT_DIR+'/log', epoch)
                    self.validate_model(sess)
            print "Done. Traing time: %s seconds" % (time.time() - start_time)
        writer.close()

    def validate_model(self, sess):
        # Testing model

        # softmax on y_predict outputs
        y_predict_labels = tf.nn.softmax(self._logits)
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
    model = ConvnetModel(BATCH_SIZE)
    model.build_model()
    model.load_data()
    model.train_model(EPOCH_SIZE)
    #model.validate_model()
    
