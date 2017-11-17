import tensorflow as tf
from load_data import *

final = False

EPOCHS = 10
BATCH_SIZE = 50
dropout = 0.95
n_classes = len(set(y_train))
(width, height, n_colors) = X_train.shape[1:4]

# Store layers weight & bias
weights = {
    # 3 * 3 filter, 1 input, 6 output
    'wc1': tf.Variable(tf.random_normal([3, 3, n_colors, 6], mean = 0, stddev = 0.1)),#CHANGED
    # 5 * 5 filter, 6 input, 16 output
    'wc2': tf.Variable(tf.random_normal([5, 5, 6, 16], mean = 0, stddev = 0.1)),
    # 5*5*16 inputs, 120 outputs
    'wf1': tf.Variable(tf.random_normal([5*5*16, 120], mean = 0, stddev = 0.1)),
    # 120 inputs, 84 outputs
    'wf2': tf.Variable(tf.random_normal([120, 84], mean = 0, stddev = 0.1)),
    # 84 inputs, 10 outputs
    'out': tf.Variable(tf.random_normal([84, n_classes], mean = 0, stddev = 0.1))}

biases = {
    # 6 outputs from layer
    'bc1': tf.Variable(tf.zeros([6])),
    # 16 outputs from layer
    'bc2': tf.Variable(tf.zeros([16])),
    # 120 outputs from FC1
    'bf1': tf.Variable(tf.zeros([120])),
    # 84 outputs from FC2
    'bf2': tf.Variable(tf.zeros([84])),
    # 10 outputs from FC3
    'out': tf.Variable(tf.zeros([n_classes]))}

# MNIST consists of 28x28x1, grayscale images
x = tf.placeholder(tf.float32, (None, width, height, n_colors))#CHANGED
# Classify over 10 digits 0-9
y = tf.placeholder(tf.int32, (None))#CHANGED
one_hot_y = tf.one_hot(y, depth = n_classes)#CHANGED