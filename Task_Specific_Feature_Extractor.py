import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os

def x_ray_task_specific(Model_save_directory,train_embed,test_embed,train_labels,test_labels):

	x_task = keras.Sequential()
	x_task.add(keras.layers.BatchNormalization())
	x_task.add(keras.layers.Dense(1120))
	x_task.add(keras.layers.LeakyReLU())
	x_task.add(keras.layers.Dense(960))
	x_task.add(keras.layers.LeakyReLU())
	x_task.add(keras.layers.Dense(800))
	x_task.add(keras.layers.LeakyReLU())
	x_task.add(keras.layers.Dense(640))
	x_task.add(keras.layers.LeakyReLU())
	x_task.add(keras.layers.BatchNormalization())
	x_task.add(keras.layers.Dense(1))
	save_callback = keras.callbacks.ModelCheckpoint(Model_save_directory,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
	x_task.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy',keras.metrics.AUC(from_logits=True)])
	hist = x_task.fit(x=train_embed,y=train_labels,epochs=100,verbose=1,validation_data=(test_embed,test_labels),use_multiprocessing=True,callbacks=save_callback)

def ct_scan_task_specific(Model_save_directory,train_embed,test_embed,train_labels,test_labels):

	ct_task = keras.Sequential()
	ct_task.add(keras.layers.BatchNormalization())
	ct_task.add(keras.layers.Dense(1120))
	ct_task.add(keras.layers.LeakyReLU(0.5))
	ct_task.add(keras.layers.Dropout(0.45))
	ct_task.add(keras.layers.Dense(960))
	ct_task.add(keras.layers.LeakyReLU(0.5))
	ct_task.add(keras.layers.Dropout(0.45))
	ct_task.add(keras.layers.Dense(800))
	ct_task.add(keras.layers.LeakyReLU(0.5))
	ct_task.add(keras.layers.Dropout(0.45))
	ct_task.add(keras.layers.Dense(640))
	ct_task.add(keras.layers.LeakyReLU(0.5))
	ct_task.add(keras.layers.BatchNormalization())
	ct_task.add(keras.layers.Dense(1))
	save_callback = keras.callbacks.ModelCheckpoint(Model_save_directory,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
	ct_task.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy',keras.metrics.AUC(from_logits=True)])
	hist = ct_task.fit(x=train_embed,y=train_labels,epochs=100,verbose=1,validation_data=(test_embed,test_labels),use_multiprocessing=True,callbacks=save_callback)

print("Enter the Function to be implemented ?")
print("1. x_ray_task_specific ")
print("2. ct_scan_task_specific ")
option = input()
if (option == "1"):
	save_directory = input("Enter the File Path for Chest X-Ray Task Specific Module --> ")
	train_embed = np.load(input("Enter the File Path for Chest X-Ray Train Embeddings --> "))
	test_embed = np.load(input("Enter the File Path for Chest X-Ray Test Embeddings --> "))
	train_label = np.load(input("Enter the File Path for Chest X-Ray Train Labels --> "))
	test_label = np.load(input("Enter the File Path for Chest X-Ray Test Labels --> "))
	x_ray_task_specific(save_directory,train_embed,test_embed,train_label,test_label)
elif (option == "2"):
	save_directory = input("Enter the File Path for CT-Scan Task Specific Module --> ")
	train_embed = np.load(input("Enter the File Path for CT-Scan Train Embeddings --> "))
	test_embed = np.load(input("Enter the File Path for CT-Scan Test Embeddings --> "))
	train_label = np.load(input("Enter the File Path for CT-Scan Train Labels --> "))
	test_label = np.load(input("Enter the File Path for CT-Scan Test Labels --> "))
	ct_scan_task_specific(save_directory,train_embed,test_embed,train_label,test_label)
else:
	print("Inadequate Input Option")
