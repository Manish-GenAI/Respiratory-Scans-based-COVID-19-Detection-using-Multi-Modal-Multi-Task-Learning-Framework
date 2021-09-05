import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

def mlp_model():
	mlp = keras.Sequential()
	mlp.add(keras.layers.InputLayer(input_shape=1280))
	mlp.add(keras.layers.Dense(960))
	mlp.add(keras.layers.LeakyReLU())
	mlp.add(keras.layers.Dense(800))
	mlp.add(keras.layers.LeakyReLU())
	mlp.add(keras.layers.Dense(640))
	mlp.add(keras.layers.LeakyReLU())
	mlp.add(keras.layers.Dense(1,activation='sigmoid'))
	return mlp

def mlp_train(Model_save_Directory,x_train_embed,x_test_embed,ct_train_embed,ct_test_embed,train_labels,test_labels):
	train = np.concatenate((x_train_embed,ct_train_embed),axis=0)
	test = np.concatenate((x_test_embed,ct_test_embed),axis=0)
	mlp = mlp_model()
	mlp.compile(optimizer='Adam',loss=keras.losses.BinaryCrossentropy(from_logits=False),metrics=['accuracy',keras.metrics.AUC(from_logits=False)])
	save = keras.callbacks.ModelCheckpoint(Model_save_Directory,monitor='val_accuracy',mode='max',save_best_only=True,verbose=1)
	hist = mlp.fit(x=train,y=train_labels,validation_data=(test,test_labels),verbose=1,epochs=100,batch_size=32,use_multiprocessing=True,callbacks=save)
