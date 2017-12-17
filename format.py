import cv2
from matplotlib import pyplot as plt
import numpy as np
import PIL
import os
import csv

folder = '.\\German Road Signs\\'
folder_32 = '.\\German Road Signs\\32x\\'

X_int = np.empty((5, 32, 32, 3), dtype=np.float32)
y_int = np.empty((5), dtype=np.float32)

print('\nProcessing test images from internet:')
print('=====================================\n')

signnames = []
with open('signnames.csv', 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
		signnames.append(row)


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

for im in X_int:
	plt.imshow(im)
	plt.show()

y_int[0] = 12 # priority road
y_int[1] = 34 # turn left ahead
y_int[2] = 32 # end speed limits
y_int[3] = 13 # yield
y_int[4] = 11 # right of way