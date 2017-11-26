from network_description import *
from helper_functions import conv2d, maxpool2d
import tensorflow as tf


def LeNet(x, keep_prob):

    # Convolution 1 - 32*32*1 to 30*30*6. Dimension of filter is 3*3 and the padding is VALID
    with tf.name_scope("convolution"):
        conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides = conv_stride, padding = P)
    # Apply rectified linear activation function
    with tf.name_scope("activation"):
        actv1 = tf.nn.relu(conv1)
    # Pool 1 - 30*30*6 to 15*15*6. Dimension of filter is 2*2 and padding is VALID
    with tf.name_scope("pooling"):
        pool1 = maxpool2d(actv1, pool_dim = pool_dim, pool_stride = pool_stride,  padding = P)
    # Apply dropout after max pooling
    #with tf.name_scope("dropout"):
    #    drop1 = tf.nn.dropout(pool1, keep_prob)


    #drop1 = conv2d(drop1, weights['1x1'], biases['1x1'], strides = conv_stride, padding = P)


    # Layer 2 - 15*15*6 to 14*14*16. Dimension of filter is 2*2 and the padding is VALID
    with tf.name_scope("convolution"):
        conv2 = conv2d(pool1, weights['wc2'], biases['bc2'], strides = conv_stride, padding = P)
    # Apply rectified linear activation function
    with tf.name_scope("activation"):
        actv2 = tf.nn.relu(conv2)
    # Pool 2 - 14*14*16 to 7*7*16. Dimension of filter is 2*2 and padding is VALID
    with tf.name_scope("pooling"):
        pool2 = maxpool2d(actv2, pool_dim = pool_dim, pool_stride = pool_stride, padding = P)
    # Apply dropout after max pooling
    #with tf.name_scope("dropout"):
    #    drop2 = tf.nn.dropout(pool2, keep_prob)

    #drop2 = conv2d(drop2, weights['1x1_2'], biases['1x1_2'], strides = conv_stride, padding = P)


    # Layer 2 - 15*15*6 to 14*14*16. Dimension of filter is 2*2 and the padding is VALID
    with tf.name_scope("convolution"):
        conv3 = conv2d(pool2, weights['wc3'], biases['bc3'], strides = conv_stride, padding = P)
    # Apply rectified linear activation function
    with tf.name_scope("activation"):
        actv3 = tf.nn.relu(conv3)
    # Pool 2 - 14*14*16 to 7*7*16. Dimension of filter is 2*2 and padding is VALID
    with tf.name_scope("pooling"):
        pool3 = maxpool2d(actv3, pool_dim = pool_dim, pool_stride = pool_stride, padding = P)
    # Apply dropout after max pooling
    #with tf.name_scope("dropout"):
    #    dropx = tf.nn.dropout(pool3, keep_prob)

    #dropx = conv2d(dropx, weights['1x1_3'], biases['1x1_3'], strides = conv_stride, padding = P)


    # Flatten the 3D stack of images to 1D
    flatten = tf.contrib.layers.flatten(pool3)

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

    # Output Layer - class prediction - 1024 to 10
    with tf.name_scope("out"):
        out = tf.add(tf.matmul(drop4, weights['out']), biases['out'])

    # Return the result of the last fully connected layer.
    return out, conv1, conv2, conv3, fc1, fc2