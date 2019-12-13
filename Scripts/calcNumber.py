#import the necessary packages
# 
# from imutils import contours
# import imutils
import argparse
import numpy as np
import cv2
import os

def mse(imA, imB):
	imA = cv2.resize(imA,(200,200))
	imB = cv2.resize(imB,(200,200))
	# Calculate 'Mean Squared Error' between images
	err = np.sum((imA.astype("float") - imB.astype("float")) ** 2)
	err /= float(imA.shape[0] * imA.shape[1])
	# The lower the error, the more similar the images are
	return err
'''
def getNumber(image):
	# Resize image to the size of the other images (200x200)
	image = cv2.resize(image,(200,200))
	# Default values
	bestMSE = 9999999
	bestSSIM = -1
	numVal = -1
	# For every file in BoardCoins, load the image and compare
	# with this one, storing the lowest index
	for filename in os.listdir('../BoardCoins/'):
		newfilename = '../BoardCoins/' + filename
		compIm = cv2.imread(newfilename)
		meansquare = mse(image,compIm)
		ss = measure.compare_ssim(image,compIm,multichannel=True)		
		#if(meansquare < bestMSE):
		#	bestMSE = meansquare
		#	filetrunc = filename[:-4]
		#	numVal = int(filetrunc)
		if(ss > bestSSIM):
			bestSSIM = ss
			filetrunc = filename[:-4]
			numVal = int(filetrunc)
	return numVal
'''
def main():
	image = cv2.imread('../two.jpg')
	yeet = getNumber(image)
	print(yeet)

if __name__ == "__main__":
	main()
