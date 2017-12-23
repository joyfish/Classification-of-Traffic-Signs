from matplotlib import pyplot as plt
import numpy as np


def hist(labels, title):

	""" Function to plot a histogram of the number of occurrances
	of each label in the training dataset """

	plt.hist(labels, bins=np.arange(44)-0.5)
	plt.ylabel('Number of Occurrances')
	plt.title(title)

	locs, labels = plt.xticks()
	plt.xticks(range(0,43), range(0,43))

def plot_input_channels(X, i):

    """ Function to plot one image from each input channel """

    D ={0 : 'Red Channel',
        1 : 'Green Channel',
        2 : 'Blue Channel'}
        #3 : 'Grey Channel',
        #4 : 'Red Flipped',
        #5 : 'Green Flipped',
        #6 : 'Blue Flipped',
        #7 : 'Edge Detection',
        #8 : '45 Deg Right',
        #9 : '45 Deg Left'}

    C ={0 : 'Reds',
        1 : 'Greens',
        2 : 'Blues'}
        #3 : 'Greys',
        #4 : 'Reds',
        #5 : 'Greens',
        #6 : 'Blues',
        #7 : 'Greys_r',
        #8 : 'Greys_r',
        #9 : 'Greys_r'}
    fig = plt.figure()
    for idx in range(0, 3):
        plt.subplot(1, 3, idx + 1)
        plt.imshow(X[i, :, :, idx], cmap = C[idx])
        plt.title(D[idx], size = 'small')
    plt.tight_layout()
    plt.show()
    #savefig('input_channels.png')


def plot_validation(val_correct, y, string = 'Validation'):

	""" Function to identify the images in the validation set which
	were not classified correctly """

	x_axis = np.linspace(0,len(val_correct),len(val_correct))
	plt.plot(x_axis, val_correct)
	plt.ylabel('Classified In/Correctly')
	plt.xlabel('Example Number')
	plt.title('Classification of ' + string + ' Images v. Order of Label')
	plt.grid()
	plt.show()

	correct, incorrect = [], []

	for i, flag in enumerate(val_correct):
		if flag == 1:
			correct.append(y[i])
		else:
			incorrect.append(y[i])

	print('Incorrect: {}'.format(incorrect))

	labels = {
		0 : correct,
		1 : incorrect
	}
	titles = {
		0 : 'Correctly Classified Examples',
		1 : 'Incorrectly Classified Examples'
	}

	for i in range(0, 2):
		plt.subplot(2, 1, i + 1)
		plt.grid()
		hist(labels[i], titles[i])
	plt.show()