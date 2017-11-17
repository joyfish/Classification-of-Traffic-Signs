import tensorflow as tf
import numpy as np

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

def norm(img):

    greyscale = False

    img_a = img[:, :, :, 0] # Red
    img_b = img[:, :, :, 1] # Green
    img_c = img[:, :, :, 2] # Blue

    # normalizing per channel data:
    img_a = (img_a - np.min(img_a)) / (np.max(img_a) - np.min(img_a))
    img_b = (img_b - np.min(img_b)) / (np.max(img_b) - np.min(img_b))
    img_c = (img_c - np.min(img_c)) / (np.max(img_c) - np.min(img_c))

    if greyscale:
        # 1 channel (greyscale)
        img_ret = np.empty((img.shape[0], img.shape[1], img.shape[2], 1), dtype=np.float32)
        img_ret[:, :, :, 0] = 0.2989 * img_a + 0.5870 * img_b + 0.1140 * img_c

    else:
        # putting the 3 channels back together:
        img_ret = np.empty(img.shape , dtype=np.float32)
        img_ret[:, :, :, 0] = img_a
        img_ret[:, :, :, 1] = img_b
        img_ret[:, :, :, 2] = img_c


    return img_ret
