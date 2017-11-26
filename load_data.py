from helper_functions import norm_add_greyscale, canny_and_flip
from plot import hist, plot_input_channels
from matplotlib import pyplot as plt
import tensorflow as tf
from format import *
import pickle

greyscale = False

with tf.name_scope("load"):

	training_file = 'traffic-signs-data/train.p'
	validation_file= 'traffic-signs-data/valid.p'
	testing_file = 'traffic-signs-data/test.p'

	with open(training_file, mode='rb') as f:
	    train = pickle.load(f)
	with open(validation_file, mode='rb') as f:
	    valid = pickle.load(f)
	with open(testing_file, mode='rb') as f:
	    test = pickle.load(f)
	    
	X_train, y_train = train['features'], train['labels']
	X_valid, y_valid = valid['features'], valid['labels']
	X_test, y_test = test['features'], test['labels']

with tf.name_scope("preprocess"):

	# Normalize RGB channels, and add greyscale channel
	(X_train, X_valid, X_test, X_int) = map(norm_add_greyscale, [X_train, X_valid, X_test, X_int])
	# Add 3 flipped RBG channels, add an edge-detected channel
	(X_train, X_valid, X_test, X_int) = map(canny_and_flip, [X_train, X_valid, X_test, X_int])

# Number of input channels after pre-processing
n_colors = X_train.shape[3]


#############################################
### Visulaize the input data in histogram ###
#############################################

labels = {
	0 : y_train,
	1 : y_valid,
	2 : y_test
}
titles = {
	0 : 'Distribution of Training Labels',
	1 : 'Distribution of Validation Labels',
	2 : 'Distribution of Test Labels'
}

# Plot distribution of labels in test, validation, training data
for i in range(0, 3):
	plt.subplot(3, 1, i + 1)
	plt.grid()
	hist(labels[i], titles[i])
plt.show()

# Plot of the 8 different channels for a sample image
plot_input_channels(X_train, 0)

#############################################
#############################################
