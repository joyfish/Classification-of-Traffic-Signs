import tensorflow as tf
from load_data import *

# Flags
final = False
l2_regularize = True

# Tunable parameters
x_bar = 0
sigma = 0.1
beta = 0.001
EPOCHS = 15
BATCH_SIZE = 20
dropout = 0.99
learning_rate = 0.001
conv_stride = 1
pool_stride = 2
pool_dim = 2

# Calculate data parameters
n_classes = len(set(y_train))
(width, height) = (int(X_train.shape[1]),int(X_train.shape[2]))

P = 'VALID'

if P == 'VALID':
    p_val = 0
else:
    p_val = 1

# Filter height and width are equal for each convolution
filters = {
    'hw_1' : 3,
    'hw_2' : 2,
    'hw_3' : 2,
}

# Fully connected layer sizes
layer_size = {
    'fc_1' : 200,
    'fc_2' : 100
}

# Output depth for convolution layers
depths_out = {
    'c_1' : 40,
    'c_2' : 100,
    'c_3' : 300
}

# Output dimension after each pooling operation
output_dims = {
    'c_1' : int((height- filters['hw_1'] + 2 * p_val)/conv_stride + 1)
}

output_dims['p_1'] = int((output_dims['c_1'] - pool_dim + 2 * p_val)/pool_stride + 1)
output_dims['c_2'] = int((output_dims['p_1'] - filters['hw_2'] + 2 * p_val)/conv_stride + 1)
output_dims['p_2'] = int((output_dims['c_2'] - pool_dim + 2 * p_val)/pool_stride + 1)
output_dims['c_3'] = int((output_dims['p_2'] - filters['hw_3'] + 2 * p_val)/conv_stride + 1)
output_dims['p_3'] = int((output_dims['c_3'] - pool_dim + 2 * p_val)/pool_stride + 1)

# Definitions for network weights
weights = {
    # Convolution 1
    'wc1': tf.Variable(tf.random_normal([filters['hw_1'], filters['hw_1'], n_colors, depths_out['c_1']], mean = x_bar, stddev = sigma)),
    # Convolution 2
    'wc2': tf.Variable(tf.random_normal([filters['hw_2'], filters['hw_2'], depths_out['c_1'], depths_out['c_2']], mean = x_bar, stddev = sigma)),
    # Convolution 3
    'wc3': tf.Variable(tf.random_normal([filters['hw_3'], filters['hw_3'], depths_out['c_2'], depths_out['c_3']], mean = x_bar, stddev = sigma)),
    # Fully Connected 1
    'wf1': tf.Variable(tf.random_normal([output_dims['p_3']*output_dims['p_3']*depths_out['c_3'], layer_size['fc_1']], mean = x_bar, stddev = sigma)),
    # Fully Connected 2
    'wf2': tf.Variable(tf.random_normal([layer_size['fc_1'], layer_size['fc_2']], mean = x_bar, stddev = sigma)),
    # Output Layer
    'out': tf.Variable(tf.random_normal([layer_size['fc_2'], n_classes], mean = x_bar, stddev = sigma))}

# Definitions for network biases
biases = {
    # Convolution 1
    'bc1': tf.Variable(tf.zeros([depths_out['c_1']])),
    # Convolution 2
    'bc2': tf.Variable(tf.zeros([depths_out['c_2']])),
    # Convolution 3
    'bc3': tf.Variable(tf.zeros([depths_out['c_3']])),
    # Fully Connected 1
    'bf1': tf.Variable(tf.zeros([layer_size['fc_1']])),
    # Fully Connected 2
    'bf2': tf.Variable(tf.zeros([layer_size['fc_2']])),
    # Output Layer
    'out': tf.Variable(tf.zeros([n_classes]))}

# Placeholder variables for feeding data into the network
x = tf.placeholder(tf.float32, (None, width, height, n_colors))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, depth = n_classes)