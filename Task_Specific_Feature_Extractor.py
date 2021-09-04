import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os

def x_ray_task_specific(Model_base_directory,train_embed_directory,test_embed_directory,train_labels,test_labels):

	x_train = np.load(train_embed_directory)
	x_test = np.load(test_embed_directory)
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
	save_callback = keras.calbacks.ModelCheckpoint(Model_base_directory,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
	x_task.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy',keras.metrics.AUC(from_logits=True)])
	hist = x_task.fit(x=x_train,y=train_labels,epochs=100,verbose=1,validation_data=(x_test,test_labels),use_multiprocessing=True,callbacks=save_callback)

def ct_scan_task_specific(Model_base_directory,train_embed_directory,test_embed_directory,train_labels,test_labels):

	ct_train = np.load(train_embed_directory)
	ct_test = np.load(test_embed_directory)
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
	save_callback = keras.calbacks.ModelCheckpoint(Model_base_directory,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
	ct_task.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy',keras.metrics.AUC(from_logits=True)])
	hist = ct_task.fit(x=ct_train,y=train_labels,epochs=100,verbose=1,validation_data=(ct_test,test_labels),use_multiprocessing=True,callbacks=save_callback)
		
