from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import os

def Classifier_Head(Model_save_Directory,train_embed,test_embed,train_labels,test_labels):
	head = keras.Sequential()
	head.add(keras.layers.InputLayer(input_shape=1280))
	head.add(keras.layers.Dense(320))
	head.add(keras.layers.LeakyReLU())
	head.add(keras.layers.Dense(80))
	head.add(keras.layers.LeakyReLU())
	head.add(keras.layers.Dense(20))
	head.add(keras.layers.LeakyReLU())
	head.add(keras.layers.Dense(1,activation='sigmoid'))
	head.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),optimizer='Adam',metrics=['accuracy',keras.metrics.AUC(from_logits=False)])
	save = keras.callbacks.ModelCheckpoint(Model_save_Directory,monitor='val_accuracy',mode='max',verbose=1,save_best_only=True)
	head.fit(x=train_embed,y=train_labels,validation_data=(test_embed,test_labels),epochs=100,batch_size=32,use_multiprocessing=True,callbacks=save)

def Final_Embeddings(shared_model,x_task_model,ct_task_model,shared_x_ray_train_input,shared_x_ray_test_input,shared_ct_train_input,shared_ct_test_input,task_x_ray_train_input,task_x_ray_test_input,task_ct_train_input,task_ct_test_input):
	for i in x_task_model.layers[:-1]:
			task_x_ray_train_input = i(task_x_ray_train_input)
			task_x_ray_test_input = i(task_x_ray_test_input)

	for j in ct_task_model.layers[:-1]:
			task_ct_train_input = j(task_ct_train_input)
			task_ct_test_input = j(task_ct_test_input)

	for k in shared_model.layers[:-1]:
			shared_x_ray_train_input= k(shared_x_ray_train_input)
			shared_x_ray_test_input = k(shared_x_ray_test_input)
			shared_ct_train_input = k(shared_ct_train_input)
			shared_ct_test_input  = k(shared_ct_test_input)
	
	x_ray_train_embed = np.concatenate((task_x_ray_train_input,shared_x_ray_train_input),axis=1)
	x_ray_test_embed = np.concatenate((task_x_ray_test_input,shared_x_ray_test_input),axis=1)
	ct_scan_train_embed = np.concatenate((task_ct_scan_train_input,shared_ct_scan_train_input),axis=1)
	ct_scan_test_embed = np.concatenate((task_ct_scan_test_input,shared_ct_scan_test_input),axis=1)
	return x_ray_train_embed,x_ray_test_embed,ct_scan_train_embed,ct_scan_test_embed
