import cv2
from matplotlib import pyplot as plt
import numpy as np
import PIL
import os

folder = '.\\German Road Signs\\'
folder_32 = '.\\German Road Signs\\32x\\'

X_int = np.empty((5, 32, 32, 3), dtype=np.float32)
y_int = np.empty((5), dtype=np.float32)

print('\nProcessing test images from internet:')
print('=====================================\n')

for i in range(0, 5):
	loc = folder + str(i + 1) + '.jpg'
	print('Image {} has been read'.format(i+1))
	I = PIL.Image.open(loc)
	try:
		# Try to save the images in the folder
		I.resize((32, 32), PIL.Image.ANTIALIAS).save(folder_32 + str(i + 1) + '32' + '.jpg')
	except FileNotFoundError:
		# Create the folder if it does not exist yet
		os.mkdir(folder_32)
		I.resize((32, 32), PIL.Image.ANTIALIAS).save(folder_32 + str(i + 1) + '32' + '.jpg')
	print('Image {} has been re-sized to 32x32'.format(i+1))

	im = cv2.imread(folder_32 + str(i + 1) + '32' + '.jpg', 1)
	X_int[i, :, :, :] = im

y_int[0] = 13 # yield sign
y_int[1] = 23 # slippery road
y_int[2] = 40 # roundabout
y_int[3] = 14 # stop
y_int[4] = 15 # no vehicle