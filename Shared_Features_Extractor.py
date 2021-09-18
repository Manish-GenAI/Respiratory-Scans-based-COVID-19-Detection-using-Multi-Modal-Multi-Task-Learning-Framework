import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pandas as pd

def disc_model():
	disc = keras.Sequential()
	disc.add(keras.layers.InputLayer(input_shape=1280))
	disc.add(keras.layers.Dense(640))
	disc.add(keras.layers.LeakyReLU())
	disc.add(keras.layers.Dense(320))
	disc.add(keras.layers.LeakyReLU())
	disc.add(keras.layers.Dense(160))
	disc.add(keras.layers.LeakyReLU())
	disc.add(keras.layers.Dense(1))
	return disc

def disc_loss_func(actual_embed,gen_embed,train_labels,invert_labels):
	actual_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=train_labels, logits=actual_embed)
	feature_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=invert_labels, logits=gen_embed)
	total_loss = actual_loss + feature_loss
	return total_loss

def gen_model():
	gen = keras.Sequential()
	gen.add(keras.layers.InputLayer(input_shape=1280))
	gen.add(keras.layers.Dense(960))
	gen.add(keras.layers.LeakyReLU())
	gen.add(keras.layers.Dense(640))
	gen.add(keras.layers.LeakyReLU())
	gen.add(keras.layers.Dense(960))
	gen.add(keras.layers.LeakyReLU())
	gen.add(keras.layers.Dense(1280))
	gen.add(keras.layers.LeakyReLU())
	return gen


def gen_loss_func(gen_embed,invert_labels):
	return tf.nn.sigmoid_cross_entropy_with_logits(labels = invert_labels, logits = gen_embed)

def train_step(actual_embed,old_gen_loss,old_disc_loss,train_labels,invert_labels):
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		features = gen(actual_embed, training=True)
		real_output = disc(actual_embed, training=True)
		features_output = disc(features, training=True)
		gen_loss = gen_loss_func(features_output,invert_labels)
		disc_loss = disc_loss_func(real_output,features_output,train_labels,invert_labels)
	gradients_of_generator = gen_tape.gradient(gen_loss, gen.variables)
	gen_optim.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
	gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.variables)
	disc_optim.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
	return gen_loss,disc_loss

def fine_tune(Model_save_Directory,model_gen,x_train_embed,x_test_embed,ct_train_embed,ct_test_embed):
	x_train_labels = np.ones((x_train_embed.shape[0],))
	ct_train_labels = np.zeros((ct_train_embed.shape[0],))
	x_test_labels = np.ones((x_test_embed.shape[0],))
	ct_test_labels = np.zeros((ct_test_embed.shape[0],))
	train_labels = np.concatenate((x_train_labels,ct_train_labels),axis=0)
	test_labels = np.concatenate((x_test_labels,ct_test_labels),axis=0)
	train_embed = np.concatenate((x_train_embed,ct_train_embed),axis=0)
	test_embed = np.concatenate((x_test_embed,ct_test_embed),axis=0)
	model_gen.trainable=False
	model = keras.Sequential()
	model.add(model_gen)
	model.add(keras.layers.LeakyReLU())
	model.add(keras.layers.Dense(640))
	model.add(keras.layers.LeakyReLU())
	model.add(keras.layers.Dense(1,activation='sigmoid'))
	model.compile(optimizer='Adam',loss=keras.losses.BinaryCrossentropy(from_logits=False),metrics=['accuracy',keras.metrics.AUC(from_logits=False)])
	save = keras.callbacks.ModelCheckpoint(Model_save_Directory,monitor='val_accuracy',mode='max',save_best_only=True,verbose=1)
	hist = model.fit(x=train_embed,y=train_labels,validation_data=(test_embed,test_labels),epochs=100,verbose=1,use_multiprocessing=True,callbacks=save)
 
disc_optim = tf.optimizers.Adam(0.00001)
gen_optim = tf.optimizers.Adam(0.00001)
print("Enter the Function to be implemented ?")
print("1. Adversarial Training for Shared Features Module ")
print("2. Fine Tuning the Adversarially Trained Shared Features Module ")
option = input()
if (option == "1"):
	x_ray_train = np.load(input("Enter the File Path for Chest X-Ray Train Embeddings"))
	ct_scan_train = np.load(input("Enter the File Path for CT-Scan Train Embeddings"))
	train = np.concatenate((x,ct),axis=0)
	x_label = [1.0]*x_ray_train.shape[0]
	ct_label = [0.0]*ct_scan_train.shape[0]
	train_label = x_label + ct_label
	train_label = np.array(train_label).astype('float32')
	invert_label = np.array(ct_label+x_label).astype('float32')
	gen = gen_model()
	disc = disc_model()
	epochs = 10
	gen_loss = 500
	disc_loss = 500
	for epoch in range(epochs):
	gen_loss,disc_loss = train_step(train,gen_loss,disc_loss,train_label.reshape((train_label.shape[0],1)),invert_label.reshape((invert_label.shape[0],1)))
 
elif ( option == "2"):
	x_ray_train = np.load(input("Enter the File Path for Chest X-Ray Train Embeddings"))
	x_ray_test = np.load(input("Enter the File Path for Chest X-Ray Test Embeddings"))
	ct_scan_train = np.load(input("Enter the File Path for CT-Scan Train Embeddings"))
	ct_scan_test = np.load(input("Enter the File Path for CT-Scan Test Embeddings"))
	save_directory = input("Enter the File Path to save the Adversarially fine-tuned Shared Features Module")
	model_gen = keras.models.load_model(input("Enter the File Path for the Adversarially trained Shared Features Module"))
	fine_tune(save_directory,model_gen,x_ray_train,x_ray_test,ct_scan_train,ct_scan_test)
else:
	print("Inadequate Input Option")
