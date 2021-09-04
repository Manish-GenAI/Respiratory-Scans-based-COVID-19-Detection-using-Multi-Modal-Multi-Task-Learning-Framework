import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFilter

def Sample_Processing(x_ray_directory,ct_scan_directory):

	final_x = cv2.imread(x_ray_directory,0)
	final_ct = cv2.imread(ct_scan_directory,0)

	#Thresholding Methods

	ret,thresh1 = cv2.threshold(final_ct,100,255,cv2.THRESH_BINARY)
	ret,thresh2 = cv2.threshold(final_ct,100,255,cv2.THRESH_BINARY_INV)
	ret,thresh3 = cv2.threshold(final_ct,100,255,cv2.THRESH_TRUNC)
	ret,thresh4 = cv2.threshold(final_ct,100,255,cv2.THRESH_TOZERO)
	ret,thresh5 = cv2.threshold(final_ct,100,255,cv2.THRESH_TOZERO_INV)
	th2 = cv2.adaptiveThreshold(final_ct,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(final_ct,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	ret,th4 = cv2.threshold(final_ct,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	ret,thresh1 = cv2.threshold(final_x,100,255,cv2.THRESH_BINARY)
	ret,thresh2 = cv2.threshold(final_x,100,255,cv2.THRESH_BINARY_INV)
	ret,thresh3 = cv2.threshold(final_x,100,255,cv2.THRESH_TRUNC)
	ret,thresh4 = cv2.threshold(final_x,100,255,cv2.THRESH_TOZERO)
	ret,thresh5 = cv2.threshold(final_x,100,255,cv2.THRESH_TOZERO_INV)
	th2 = cv2.adaptiveThreshold(final_x,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(final_x,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	ret,th4 = cv2.threshold(final_x,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	#Smoothening Filters

	img1 = cv2.blur(final_ct,(3,3))
	img2 = cv2.medianBlur(final_ct,3)
	img3 = cv2.GaussianBlur(final_ct,(3,3),0)
	img4 = cv2.bilateralFilter(final_ct,9,75,75)

	img1 = cv2.blur(final_x,(3,3))
	img2 = cv2.medianBlur(final_x,3)
	img3 = cv2.GaussianBlur(final_x,(3,3),0)
	img4 = cv2.bilateralFilter(final_x,9,75,75)

	#Morphological Operations

	ker_ct = np.ones((2,2))
	img1 = cv2.erode(final_ct,ker_ct,iterations=1)
	img2 = cv2.dilate(final_ct,ker_ct,iterations=1)
	img3 = cv2.morphologyEx(final_ct, cv2.MORPH_OPEN, ker_ct)
	img4 = cv2.morphologyEx(final_ct, cv2.MORPH_CLOSE, ker_ct)
	img5 = cv2.morphologyEx(final_ct, cv2.MORPH_GRADIENT, ker_ct)
	img6 = cv2.morphologyEx(final_ct, cv2.MORPH_TOPHAT, ker_ct)
	img7 = cv2.morphologyEx(final_ct, cv2.MORPH_BLACKHAT, ker_ct)

	ker_x = np.ones((2,2))
	img1 = cv2.erode(final_x,ker_x,iterations=1)
	img2 = cv2.dilate(final_x,ker_x,iterations=1)
	img3 = cv2.morphologyEx(final_x, cv2.MORPH_OPEN, ker_x)
	img4 = cv2.morphologyEx(final_x, cv2.MORPH_CLOSE, ker_x)
	img5 = cv2.morphologyEx(final_x, cv2.MORPH_GRADIENT, ker_x)
	img6 = cv2.morphologyEx(final_x, cv2.MORPH_TOPHAT, ker_x)
	img7 = cv2.morphologyEx(final_x, cv2.MORPH_BLACKHAT, ker_x)

	#Gradient Tools

	img1 = cv2.Laplacian(final_ct,cv2.CV_64F)
	img2 = cv2.Sobel(final_ct,cv2.CV_64F,1,0,ksize=5)
	img3 = cv2.Sobel(final_ct,cv2.CV_64F,0,1,ksize=5)
	img4 = cv2.Canny(final_ct,100,100)
	image = Image.fromarray(final_ct.astype('uint8'))
	img5 = image.filter(ImageFilter.UnsharpMask(radius=3, percent=150))

	img1 = cv2.Laplacian(final_x,cv2.CV_64F)
	img2 = cv2.Sobel(final_x,cv2.CV_64F,1,0,ksize=5)
	img3 = cv2.Sobel(final_x,cv2.CV_64F,0,1,ksize=5)
	img4 = cv2.Canny(final_x,100,100)
	image = Image.fromarray(final_x.astype('uint8'))
	img5 = image.filter(ImageFilter.UnsharpMask(radius=3, percent=150))

	#Histogram Equalizations

	img1 = cv2.equalizeHist(final_ct)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img2 = clahe.apply(final_ct)

	img1 = cv2.equalizeHist(final_x)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img2 = clahe.apply(final_x)

def Dataset_Processing(x_base_directory,ct_base_directory):

	for i in tqdm(os.listdir(x_base_directory)):
		img_x = cv2.imread(os.path.join(x_base_directory,i),0)
		clahe_x = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
		final_x = clahe_x.apply(img_x)
		final_x = cv2.resize(final_x,(224,224))
		cv2.imwrite(os.path.join(x_base_directory,i),final_x)

	for j in tqdm(os.listdir(ct_base_directory)):
		img_ct = cv2.imread(os.path.join(ct_base_directory,j),0)
		clahe_ct = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
		final_ct = clahe_ct.apply(img_ct)
		final_ct = cv2.resize(final_ct,(224,224))
		cv2.imwrite(os.path.join(ct_base_directory,i),final_ct)
