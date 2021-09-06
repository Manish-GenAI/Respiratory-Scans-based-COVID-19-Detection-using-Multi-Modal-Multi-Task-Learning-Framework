import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

def Embedding_Model(Model_save_directory,Model_Class,Dataset_train_Directory,Dataset_test_Directory):
	
	if (Model_Class != keras.applications.NASNetLarge):
		ds_train = keras.preprocessing.image_dataset_from_directory(Dataset_train_Directory,
						labels = 'inferred',
						label_mode = 'binary',
						image_size=(224,224),
						batch_size = 32)
		ds_validation = keras.preprocessing.image_dataset_from_directory(Dataset_test_Directory,
						labels = 'inferred',
						label_mode = 'binary',
						image_size=(224,224),
						batch_size = 32)
		base_model = Model_Class(input_shape=(224,224,3),include_top=False,weights='imagenet')
		for layer in base_model.layers:
			layer.trainable = False
		model = keras.Sequential()
		model.add(base_model)
		model.add(keras.layers.GlobalAveragePooling2D())
		model.add(keras.layers.Dense(1280))
		model.add(keras.layers.LeakyReLU())
		model.add(keras.layers.Dense(1))
		save_callback = keras.callbacks.ModelCheckpoint(Model_save_directory,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
		model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy',keras.metrics.AUC(from_logits=True)])
		hist = model.fit(ds_train,epochs=100,verbose=1,validation_data=ds_validation,use_multiprocessing=True,callbacks=save_callback)

	else:
		ds_train = keras.preprocessing.image_dataset_from_directory(Dataset_train_Directory,
						labels = 'inferred',
						label_mode = 'binary',
						image_size=(331,331),
						batch_size = 32)
		ds_validation = keras.preprocessing.image_dataset_from_directory(Dataset_test_Directory,
						labels = 'inferred',
						label_mode = 'binary',
						image_size=(331,331),
						batch_size = 32)
		base_model = Model_Class(input_shape=(331,331,3),include_top=False,weights='imagenet')
		for layer in base_model.layers:
			layer.trainable = False
		model = keras.Sequential()
		model.add(base_model)
		model.add(keras.layers.GlobalAveragePooling2D())
		model.add(keras.layers.Dense(1280))
		model.add(keras.layers.LeakyReLU())
		model.add(keras.layers.Dense(1))
		save_callback = keras.callbacks.ModelCheckpoint(Model_save_directory,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
		model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy',keras.metrics.AUC(from_logits=True)])
		hist = model.fit(ds_train,epochs=100,verbose=1,validation_data=ds_validation,use_multiprocessing=True,callbacks=save_callback)

def Embedding_Save(Embed_save_Directory,Trained_Model_Class,Dataset_Directory,Image_Size):
	if (Image_Size != 331):
		x_train = keras.preprocessing.image_dataset_from_directory(Dataset_Directory,label_mode=None,labels=None,batch_size=32,image_size=(Image_Size,Image_Size)))
	else:
		x_train = keras.preprocessing.image_dataset_from_directory(Dataset_Directory,label_mode=None,labels=None,batch_size=32,image_size=(331,331)))
	Trained_Model_Class.trainable=False
	Trained_Model_Class.pop()
	np_embed = []
	for x in x_train:
		interm = Trained_Model_Class(x).numpy().tolist()
		for inte in interm:
			np_embed.append(inte)
	np_embed = np.array(np_embed)
	np.save(Embed_save_Directory,np_embed)
