from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import os

def Classifier_Head(Model_save_Directory,task_embedtr,shared_embedtr,task_embedte,shared_embedte,labeltr,labelte):
  train = np.concatenate((task_embedtr,shared_embedtr),axis=1)
  test = np.concatenate((task_embedte,shared_embedte),axis=1)
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
  head.fit(x=train,y=labeltr,validation_data=(test,labelte),epochs=100,batch_size=32,use_multiprocessing=True,callbacks=save)

def Final_Embeddings(shared_x_ray_train_embed,shared_x_ray_test_embed,shared_ct_train_embed,shared_ct_test_embed,task_x_ray_train_embed,task_x_ray_test_embed,task_ct_train_embed,task_ct_test_embed,shared_model,x_task_model,ct_task_model):
	for i in x_task_model.layers[:-1]:
  		task_x_ray_train_embed = i(task_x_ray_train_embed)
  		task_x_ray_test_embed = i(task_x_ray_test_embed)

	for j in ct_task_model.layers[:-1]:
  		task_ct_train_embed = j(task_ct_train_embed)
  		task_ct_test_embed = j(task_ct_test_embed)

	for k in shared_model.layers[:-1]:
  		shared_x_ray_train_embed= k(shared_x_ray_train_embed)
  		shared_x_ray_test_embed = k(shared_x_ray_test_embed)
  		shared_ct_train_embed = k(shared_ct_train_embed)
  		shared_ct_test_embed  = k(shared_ct_test_embed)

  	return shared_x_ray_train_embed,shared_x_ray_test_embed,shared_ct_train_embed,shared_ct_test_embed,task_x_ray_train_embed,task_x_ray_test_embed,task_ct_train_embed,task_ct_test_embed