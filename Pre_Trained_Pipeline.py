import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import cv2
import matplotlib.pyplot as plt

def Pre_Process(x,ct):
	clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
	clahe.apply(x)
	clahe.apply(ct)
	ker = np.ones((2,2))
	ct = cv2.morphologyEx(ct, cv2.MORPH_OPEN, ker)
	x = cv2.resize(x,(224,224))
	ct = cv2.resize(ct,(224,224))
	x = cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)
	ct = cv2.cvtColor(ct,cv2.COLOR_GRAY2RGB)
	return x.reshape((1,224,224,3)),ct.reshape((1,224,224,3))

def Task_Specific_Embed(Save_Folder_Directory,x,ct):
	x_model = keras.models.load_model(Save_Folder_Directory+'/X_Embed_Task_EfficientNetB0.h5',compile=False)
	ct_model = keras.models.load_model(Save_Folder_Directory+'/CT_Embed_Task_EfficientNetB1.h5',compile=False)
	task_x = keras.models.load_mode(Save_Folder_Directory+'X_Ray_Task.h5',compile=False)
	task_ct = keras.models.load_model(Save_Folder_Directory+'/CT_Scan_Task.h5',compile=False)

	x_layers = x_model.layers[:-1]
	ct_layers = ct_model.layers[:-1]
	task_x_layers = task_x.layers[2:-1]
	task_ct_layers = task_ct.layers[1:-1]

	for i in x_layers:
		x = i(x)
	for j in ct_layers:
		ct = j(ct)
	for k in task_x_layers:
		x = k(x)
	for l in task_ct_layers:
		ct = l(ct)
	return x,ct

def Shared_Feature_Embed(Save_Folder_Directory,x,ct):
	x_model = keras.models.load_model(Save_Folder_Directory+'/X_Embed_Shared_ResNet50.h5',compile=False)
	ct_model = keras.models.load_model(Save_Folder_Directory+'/CT_Embed_Shared_ResNet50V2.h5',compile=False)
	shared_model = keras.models.load_model(Save_Folder_Directory+'/Shared_Model.h5',compile=False)
	x_layers = x_model.layers[:-1]
	ct_layers = ct_model.layers[:-1]
	sh_layers = shared_model.layers[:-1]
	for i in x_layers:
		x = i(x)
	for j in ct_layers:
		ct = j(ct)
	for k in sh_layers:
		x = k(x)
		ct = k(ct)
	return x,ct

def Classification(Save_Folder_Directory,x,ct):
	classifier_x = keras.models.load_model(Save_Folder_Directory+'/Classifier_Head_X.h5',compile=False)
	classifier_ct = keras.models.load_model(Save_Folder_Directory+'/Classifier_Head_CT.h5',compile=False)
	ta_x,ta_ct = Task_Specific_Embed(Save_Folder_Directory,x,ct)
	sh_x,sh_ct = Shared_Feature_Embed(Save_Folder_Directory,x,ct)
	Feature_x = np.concatenate((ta_x,sh_x),axis=1)
	Feature_ct = np.concatenate((ta_ct,sh_ct),axis=1)
	result_x = classifier_x.predict(Feature_x)
	result_ct = classifier_ct.predict(Feature_ct)
	for result in [result_x,result_ct]:
		print("Probability :-", result)
		if result>=0.5:
		print("Prediction :- COVID")
		else:
		print("Prediction :- Non-COVID")
	print(" ")

def Test_Run(Save_Folder_Directory,X_Ray_Grey_Image,CT_Scan_Grey_Image):
	X_Ray_Image,CT_Scan_Image = Pre_Process(X_Ray_Grey_Image,CT_Scan_Grey_Image)
	Classification(Save_Folder_Directory,X_Ray_Image,CT_Scan_Image)

x_ray = cv2.imread(input("Enter the File Path for Chest X-Ray Input Image --> "),0)
ct_scan = cv2.imread(input("Enter the File Path for CT-Scan Input Image --> "),0)
root_directory = input("Enter the Root Folder Directory Path for Each and Every saved Components of the Pre-Trained Pipeline --> ")
Test_Run(root_directory,x_ray,ct_scan)
