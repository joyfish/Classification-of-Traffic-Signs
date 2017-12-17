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

# True  = load old data
preprocs = False
import time

with tf.name_scope("preprocess"):

	# Normalize RGB channels, and add greyscale channel
	(X_train, X_valid, X_test, X_int) = map(norm_add_greyscale, [X_train, X_valid, X_test, X_int])


	# Add 3 flipped RBG channels, add an edge-detected channel
	#(X_train, X_valid, X_test, X_int) = map(canny_and_flip, [X_train, X_valid, X_test, X_int])
	
	if not preprocs:
		# Preprocess and save if not yet done
		(X_train, y_train) = canny_and_flip(X_train, y_train)
		with open('preprocs_train_3channel.pkl', 'wb') as f:
		    pickle.dump([X_train, y_train], f)
	else:
		# Load if already preprocessed
		t = time.time()
		with open('preprocs_train.pkl', 'rb') as f:
		    X_train, y_train = pickle.load(f)
		print('{} seconds to load processed data'.format(time.time() - t))




# Number of input channels after pre-processing
#n_colors = X_train.shape[3]
n_colors = 3
print('colors: {}'.format(n_colors))

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
#savefig('histogram_input_data.png')

# Plot of the 8 different channels for a sample image
#plot_input_channels(X_train, 0)

#############################################
#############################################


#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
#X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 1)
