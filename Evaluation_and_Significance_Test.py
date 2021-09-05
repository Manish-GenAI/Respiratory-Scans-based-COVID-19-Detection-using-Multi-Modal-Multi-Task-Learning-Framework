import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import f_oneway as anova

def Evaluation_metrics(prob_list,labels,threshold=0.5):
	for i in range(0,len(prob_list)):
		if (prob_list[i] >= threshold):
			prob_list[i] = 1
		else:
			prob_list[i] = 0
	con = confusion_matrix(labels,prob_list)
	print("Confusion Matrix")
	print(con)
	print(" ")
	sen = con[0][0]/(con[0][0]+con[0][1])
	spe = con[1][1]/(con[1][1]+con[1][0])
	f1 = con[0][0] / (con[0][0] + (0.5*(con[0][1]+con[1][0])))
	acc = (con[0][0]+con[1][1])/(con[0][0]+con[1][0]+con[0][1]+con[1][1])
	classwise_acc = 0.5*((con[0][0]/(con[0][0]+con[1][0]))+(con[1][1]/(con[1][1]+con[0][1])))
	print("Accuracy (As a whole Data) = "+str(acc*100)+" %")
	print("Classwise Mean Accuracy = "+str(classwise_acc*100)+" %")
	print("Sensitivity = "+str(sen*100)+" %")
	print("Specificity = "+str(spe*100)+" %")
	print("F1 Score = "+str(f1*100)+" %")

def Significance_Test(Save_Directory,task_train_embed,shared_train_embed,task_test_embed,shared_test_embed,train_labels,test_labels):
	x_train = np.concatenate((task_train_embed,shared_train_embed),axis=1)
	x_test = np.concatenate((task_test_embed,shared_test_embed),axis=1)
	for i in range(1,10):
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
		save = keras.callbacks.ModelCheckpoint(Save_Directory+'_'+str(i)+'.h5',monitor='val_accuracy',mode='max',verbose=1,save_best_only=True)
		head.fit(x=x_train,y=train_labels,validation_data=(x_test,test_labels),epochs=100,batch_size=32,use_multiprocessing=True,callbacks=save)
	print(" ")
	for j in range(1,10):
		te = keras.models.load_model(Save_Directory+'_'+str(j)+'.h5')
		te.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),optimizer='Adam',metrics=['accuracy',keras.metrics.AUC(from_logits=False)])
		y_metric = te.evaluate(x=x_test,y=test_labels,batch_size=32,use_multiprocessing=True,verbose=1)
		y_pred = te.predict(x_test)
		np.save(Save_Directory+'_'+str(j)+'.npy',y_pred)
	print(" ")
	prob_list=[np.load(Save_Directory+'_'+str(i)+'.npy') for i in range(1,10)]
	f,p = anova(prob_list[0],prob_list[1],prob_list[2],prob_list[3],prob_list[4],prob_list[5],prob_list[6],prob_list[7],prob_list[8],prob_list[9])
	print("p-value for significance is: ", p)
	if p<0.05:
		print("reject null hypothesis")
	else:
		print("accept null hypothesis")
