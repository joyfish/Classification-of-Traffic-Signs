import tensorflow as tf
import numpy as np
import cv2
import scipy

def conv2d(x, W, b, strides = 1, padding = 'SAME'):
    """
    Function to apply convolution using specified stride and padding
    """

    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return x

def maxpool2d(x, pool_dim, pool_stride,  padding = 'SAME'):
    """
    Function to apply max pooling with specified vert/horizontal stride
    """

    return tf.nn.max_pool(
        x,
        ksize=[1, pool_dim, pool_dim, 1],
        strides=[1, pool_stride, pool_stride, 1],
        padding= padding)

def norm_add_greyscale(img):

    """ Function to normalize the input image
    (per input channel), and to add a 4th channel
    (greyscale) """

    luminosity = False

    img_a = img[:, :, :, 0] # Red
    img_b = img[:, :, :, 1] # Green
    img_c = img[:, :, :, 2] # Blue

    # normalizing per channel data:
    img_a = (img_a - np.min(img_a)) / (np.max(img_a) - np.min(img_a))
    img_b = (img_b - np.min(img_b)) / (np.max(img_b) - np.min(img_b))
    img_c = (img_c - np.min(img_c)) / (np.max(img_c) - np.min(img_c))

    # putting the 3 channels back together:
    img_ret = np.empty((img.shape[0], img.shape[1], img.shape[2], 3), dtype=np.float32)

    img_ret[:, :, :, 0] = img_a
    img_ret[:, :, :, 1] = img_b
    img_ret[:, :, :, 2] = img_c

    return img_ret

def canny_and_flip(X, y):

    """ Function to add 4 more channels to the dataset
    1. flipped red channel
    2. flipped green channel
    3. flipped blue channel
    4. canny edge detection """

    import copy
    from tqdm import tqdm

    X_temp = copy.deepcopy(X)
    y_temp = copy.deepcopy(y)
    total = X_temp.shape[0]

    print('Preprocessing training data....')

    #for i, (image, label) in enumerate(zip(tqdm(X_temp, ncols = 100), y_temp)):
        #flipped = np.flip(image, 0)
        #X[i,:,:,4] = flipped[:,:,0]
        #X[i,:,:,5] = flipped[:,:,1]
        #X[i,:,:,6] = flipped[:,:,2]

        #edges = cv2.Canny(np.uint8(image[:,:,3]*255), 30, 170)
        #X[i,:,:,7] = edges/255
        #X[i,:,:,8] = scipy.ndimage.interpolation.rotate(X[i,:,:,7], 45, reshape = False, axes=(1, 0), mode='constant')
        #X[i,:,:,9] = scipy.ndimage.interpolation.rotate(X[i,:,:,7], -45, reshape = False, axes=(1, 0), mode='constant')

        #edges = cv2.Canny(np.uint8(image*255), 30, 170)
        #X_add = edges/255
        #X_new.append(X_add), y_new.append(label)
        #X = np.insert(X, X.shape[1] , X_add, axis = 0)
        #y = np.insert(y, y.shape[0] , label, axis = 0)

        #X_add = scipy.ndimage.interpolation.rotate(image, 45, reshape = False, axes=(1, 0), mode='constant')
        #X_add = scipy.ndimage.interpolation.rotate(image, 10, reshape = False, axes=(1, 0), mode='constant')
        #X = np.insert(X, X.shape[1] , X_add, axis = 0)
        #y = np.insert(y, y.shape[0] , label, axis = 0)
        #X_add = scipy.ndimage.interpolation.rotate(image, -45, reshape = False, axes=(1, 0), mode='constant')
        #X_add = scipy.ndimage.interpolation.rotate(image, -10, reshape = False, axes=(1, 0), mode='constant')
        #X = np.insert(X, X.shape[1] , X_add, axis = 0)
        #y = np.insert(y, y.shape[0] , label, axis = 0)

    return X, y