import tensorflow as tf
from load_data import *

# Flags
pre_trained = True
internet = True
final = True
l2_regularize = True
lr_thres_1 = 0.9
lr_thres_2 = 0.94
lr_1 =  0.001
lr_2 =  0.0001
lr_3 =  0.00001


# Tunable parameters
x_bar = 0
sigma = 0.1
beta = 0.001
keep_train = 0.6
EPOCHS = 80
BATCH_SIZE = 150 #Updated - prev 200
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
    'fc_1' : 2000, #120
    'fc_2' : 900, #50
    'fc_3' : 300
}

# Output depth for convolution layers
depths_out = {
    'c_1' : 400,#1
    'c_2' : 600,#2
    'c_3' : 900#5
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



def setup_parameter_summary(kw, kb):

    ''' Method to create summary for tensorboard of the:
    1. Weights - histogram
    2. Biases - histogram
    3. Mean of weights - scalar
    4. Mean of biases - scalar '''

    tf.summary.histogram('H_{}'.format(kw),weights[kw])
    tf.summary.histogram('H_{}'.format(kb),biases[kb])
    tf.summary.scalar('H_{}_m'.format(kw),tf.reduce_mean(weights[kw]))
    tf.summary.scalar('H_{}_m'.format(kb),tf.reduce_mean(biases[kb]))

with tf.name_scope("parameters"):

    with tf.name_scope("weights"):

        # Definitions for network weights
        weights = {
            # Convolution 1
            'wc1': tf.Variable(tf.random_normal([filters['hw_1'], filters['hw_1'], n_colors, depths_out['c_1']], mean = x_bar, stddev = sigma), name = 'wc1'),
            # Convolution 2
            'wc2': tf.Variable(tf.random_normal([filters['hw_2'], filters['hw_2'], depths_out['c_1'], depths_out['c_2']], mean = x_bar, stddev = sigma), name = 'wc2'),
            # Convolution 3
            'wc3': tf.Variable(tf.random_normal([filters['hw_3'], filters['hw_3'], depths_out['c_2'], depths_out['c_3']], mean = x_bar, stddev = sigma), name = 'wc3'),
            # Fully Connected 1
            'wf1': tf.Variable(tf.random_normal([output_dims['p_3']*output_dims['p_3']*depths_out['c_3'], layer_size['fc_1']], mean = x_bar, stddev = sigma), name = 'wf1'),
            # Fully Connected 2
            'wf2': tf.Variable(tf.random_normal([layer_size['fc_1'], layer_size['fc_2']], mean = x_bar, stddev = sigma), name = 'wf2'),
            # Fully Connected 3
            'wf3': tf.Variable(tf.random_normal([layer_size['fc_2'], layer_size['fc_3']], mean = x_bar, stddev = sigma), name = 'wf3'),
            # Output Layer
            'out': tf.Variable(tf.random_normal([layer_size['fc_3'], n_classes], mean = x_bar, stddev = sigma), name = 'wout')
            }


    with tf.name_scope("biases"):
        # Definitions for network biases
        biases = {
            # Convolution 1
            'bc1': tf.Variable(tf.zeros([depths_out['c_1']]), name = 'bc1'),
            # Convolution 2
            'bc2': tf.Variable(tf.zeros([depths_out['c_2']]), name = 'bc2'),
            # Convolution 3
            'bc3': tf.Variable(tf.zeros([depths_out['c_3']]), name = 'bc3'),
            # Fully Connected 1
            'bf1': tf.Variable(tf.zeros([layer_size['fc_1']]), name = 'bf1'),
            # Fully Connected 2
            'bf2': tf.Variable(tf.zeros([layer_size['fc_2']]), name = 'bf2'),
            # Fully Connected 3
            'bf3': tf.Variable(tf.zeros([layer_size['fc_3']]), name = 'bf3'),
            # Output Layer
            'out': tf.Variable(tf.zeros([n_classes]), name = 'bout')
            }
    # Tensorboard summary of convolutional layer 1
    setup_parameter_summary(kw = 'wc1', kb = 'bc1')
    # Tensorboard summary of convolutional layer 2
    setup_parameter_summary(kw = 'wc2', kb = 'bc2')
    # Tensorboard summary of convolutional layer 3
    setup_parameter_summary(kw = 'wc3', kb = 'bc3')
    # Tensorboard summary of fully connected layer 1
    setup_parameter_summary(kw = 'wf1', kb = 'bf1')
    # Tensorboard summary of fully connected layer 2
    setup_parameter_summary(kw = 'wf2', kb = 'bf2')
    # Tensorboard summary of fully connected layer 2
    setup_parameter_summary(kw = 'wf3', kb = 'bf3')
    # Tensorboard summary of output layer
    setup_parameter_summary(kw = 'out', kb = 'out')


with tf.name_scope("placeholders"):
    # Placeholder variables for feeding data into the network
    #x = tf.placeholder(tf.float32, (None, width, height, n_colors))
    x = tf.placeholder(tf.float32, (None, width, height, 3), name = 'x')
    y = tf.placeholder(tf.int32, (None), name = 'y')

    #x_valid_p = tf.placeholder(tf.float32, (None, width, height, n_colors))
    x_valid_p = tf.placeholder(tf.float32, (None, width, height, 3), name  = 'x_valid_p')
    y_valid_p = tf.placeholder(tf.int32, (None), name = 'y_valid_p')

    length = tf.placeholder(tf.float32, name = 'length')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')

    one_hot_y = tf.one_hot(y, depth = n_classes, name = 'one_hot_y')
    one_hot_y_val = tf.one_hot(y_valid_p, depth = n_classes, name = 'one_hot_y_val')