from network_description import *
from helper_functions import conv2d, maxpool2d


def LeNet(x):

    # Convolution 1 - 32*32*1 to 30*30*6. Dimension of filter is 3*3 and the padding is VALID
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides = conv_stride, padding = P)
    # Apply rectified linear activation function
    actv1 = tf.nn.relu(conv1)
    # Pool 1 - 30*30*6 to 15*15*6. Dimension of filter is 2*2 and padding is VALID
    pool1 = maxpool2d(actv1, pool_dim = pool_dim, pool_stride = pool_stride,  padding = P)
    # Apply dropout after max pooling
    drop1 = tf.nn.dropout(pool1, dropout)

    # Layer 2 - 15*15*6 to 14*14*16. Dimension of filter is 2*2 and the padding is VALID
    conv2 = conv2d(drop1, weights['wc2'], biases['bc2'], strides = conv_stride, padding = P)
    # Apply rectified linear activation function
    actv2 = tf.nn.relu(conv2)
    # Pool 2 - 14*14*16 to 7*7*16. Dimension of filter is 2*2 and padding is VALID
    pool2 = maxpool2d(actv2, pool_dim = pool_dim, pool_stride = pool_stride, padding = P)
    # Apply dropout after max pooling
    drop2 = tf.nn.dropout(pool2, dropout)


    # Layer 2 - 15*15*6 to 14*14*16. Dimension of filter is 2*2 and the padding is VALID
    convx = conv2d(drop2, weights['wc3'], biases['bc3'], strides = conv_stride, padding = P)
    # Apply rectified linear activation function
    actvx = tf.nn.relu(convx)
    # Pool 2 - 14*14*16 to 7*7*16. Dimension of filter is 2*2 and padding is VALID
    poolx = maxpool2d(actvx, pool_dim = pool_dim, pool_stride = pool_stride, padding = P)
    # Apply dropout after max pooling
    dropx = tf.nn.dropout(poolx, dropout)

    # Flatten the 3D stack of images to 1D
    flatten = tf.contrib.layers.flatten(dropx)

    # Apply weights and biases of fully connected layer
    fc1 = tf.add(tf.matmul(flatten, weights['wf1']), biases['bf1'])
    # Apply rectified linear activation function
    actv3 = tf.nn.relu(fc1)
    # Apply dropout to the layer
    drop3 = tf.nn.dropout(actv3, dropout)

    # Apply weights and biases of fully connected layer
    fc2 = tf.add(tf.matmul(drop3, weights['wf2']), biases['bf2'])
    # Apply rectified linear activation function
    actv4 = tf.nn.relu(fc2)
    # Apply dropout to the layer
    drop4 = tf.nn.dropout(actv4, dropout)

    # Output Layer - class prediction - 1024 to 10
    out = tf.add(tf.matmul(drop4, weights['out']), biases['out'])

    # Return the result of the last fully connected layer.
    return out