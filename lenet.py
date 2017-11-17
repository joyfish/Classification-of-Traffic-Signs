from parameters import *
from helper_functions import conv2d, maxpool2d

# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.
def LeNet(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    #x = tf.reshape(x, (-1, 28, 28, 1)) #CHANGED
    # Pad 0s to 32x32. Centers the digit further.
    # Add 2 rows/columns on each side for height and width dimensions.
    #x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

    # Convolution 1 - 32*32*1 to 28*28*6. Dimension of filter is 3*3 and the padding is VALID
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], padding = 'VALID')
    # Apply rectified linear activation function
    actv1 = tf.nn.relu(conv1)
    # Pool 1 - 28*28*6 to 14*14*6. Dimension of filter is 2*2 and padding is VALID
    pool1 = maxpool2d(actv1, k=2, padding = 'VALID')
    # Apply dropout after max pooling
    drop1 = tf.nn.dropout(pool1, dropout)

    # Layer 2 - 14*14*32 to 10*10*16. Dimension of filter is 5*5 and the padding is VALID
    conv2 = conv2d(drop1, weights['wc2'], biases['bc2'], padding = 'VALID')
    # Apply rectified linear activation function
    actv2 = tf.nn.relu(conv2)
    # Pool 2 - 10*10*16 to 5*5*16. Dimension of filter is 2*2 and padding is VALID
    pool2 = maxpool2d(actv2, k=2, padding = 'VALID')
    # Apply dropout after max pooling
    drop2 = tf.nn.dropout(pool2, dropout)


    # Flatten the 3D stack of images to 1D
    flatten = tf.contrib.layers.flatten(drop2)

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