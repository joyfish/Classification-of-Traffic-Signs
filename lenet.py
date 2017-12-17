from network_description import *
from helper_functions import conv2d, maxpool2d
import tensorflow as tf


def LeNet(x, keep_prob):

    with tf.name_scope("convolution"):
    # Convolution 1 
        conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides = conv_stride, padding = P)
    # Apply rectified linear activation function
    with tf.name_scope("activation"):
        actv1 = tf.nn.relu(conv1)
    # Pool 1 
    with tf.name_scope("pooling"):
        pool1 = maxpool2d(actv1, pool_dim = pool_dim, pool_stride = pool_stride,  padding = P)
    # Apply dropout after max pooling
    with tf.name_scope("dropout"):
        drop1 = tf.nn.dropout(pool1, keep_prob)

    # Layer 2 
    with tf.name_scope("convolution"):
        conv2 = conv2d(drop1, weights['wc2'], biases['bc2'], strides = conv_stride, padding = P)
    # Apply rectified linear activation function
    with tf.name_scope("activation"):
        actv2 = tf.nn.relu(conv2)
    # Pool 2 
    with tf.name_scope("pooling"):
        pool2 = maxpool2d(actv2, pool_dim = pool_dim, pool_stride = pool_stride, padding = P)
    # Apply dropout after max pooling
    with tf.name_scope("dropout"):
        drop2 = tf.nn.dropout(pool2, keep_prob)

    # Layer 3
    with tf.name_scope("convolution"):
        conv3 = conv2d(drop2, weights['wc3'], biases['bc3'], strides = conv_stride, padding = P)
    # Apply rectified linear activation function
    with tf.name_scope("activation"):
        actv3 = tf.nn.relu(conv3)
    # Pool 3
    with tf.name_scope("pooling"):
        pool3 = maxpool2d(actv3, pool_dim = pool_dim, pool_stride = pool_stride, padding = P)
    # Apply dropout after max pooling
    with tf.name_scope("dropout"):
        dropx = tf.nn.dropout(pool3, keep_prob)

    # Flatten the 3D stack of images to 1D
    flatten = tf.contrib.layers.flatten(dropx)

    # Apply weights and biases of fully connected layer
    with tf.name_scope("fullyconnected"):
        fc1 = tf.add(tf.matmul(flatten, weights['wf1']), biases['bf1'])
    # Apply rectified linear activation function
    with tf.name_scope("activation"):
        actv3 = tf.nn.relu(fc1)
    # Apply dropout to the layer
    with tf.name_scope("dropout"):
        drop3 = tf.nn.dropout(actv3, keep_prob)

    # Apply weights and biases of fully connected layer
    with tf.name_scope("fullyconnected"):
        fc2 = tf.add(tf.matmul(drop3, weights['wf2']), biases['bf2'])
    # Apply rectified linear activation function
    with tf.name_scope("activation"):
        actv4 = tf.nn.relu(fc2)
    # Apply dropout to the layer
    with tf.name_scope("dropout"):
        drop4 = tf.nn.dropout(actv4, keep_prob)

    # Apply weights and biases of fully connected layer
    with tf.name_scope("fullyconnected"):
        fc3 = tf.add(tf.matmul(drop4, weights['wf3']), biases['bf3'])
    # Apply rectified linear activation function
    with tf.name_scope("activation"):
        actv5 = tf.nn.relu(fc3)
    # Apply dropout to the layer
    with tf.name_scope("dropout"):
        drop5 = tf.nn.dropout(actv5, keep_prob)

    # Output Layer 
    with tf.name_scope("out"):
        out = tf.add(tf.matmul(drop5, weights['out']), biases['out'], name = 'nn_out')

    # Return the result of the last fully connected layer.
    return out, conv1, conv2, conv3, fc1, fc2