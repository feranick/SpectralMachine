#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraKeras
*
* 20180119a
*
* Uses: Deep Neural Networks, TensorFlow, SVM, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
import numpy as np
import keras, sys, os.path
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, ActivityRegularization, MaxPooling1D
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
import tensorflow as tf

learnFile = sys.argv[1]
print(learnFile)

try:
    with open(learnFile, 'r') as f:
        M = np.loadtxt(f, unpack =False)
except:
    print('\033[1m' + ' Learning file not found \n' + '\033[0m')

En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
M = np.delete(M,np.s_[0:1],0)
Cl = ['{:.2f}'.format(x) for x in M[:,0]]
A = np.delete(M,np.s_[0:1],1)
learnFileRoot = os.path.splitext(learnFile)[0]

A, A_test, Cl, Cl_test = train_test_split(A, Cl, test_size=0.01, random_state=42)
En_test = En


tb_directory = "keras"
model_directory = "."
model_name = model_directory+"/keras_model.hd5"

totA = np.vstack((A, A_test))
totCl = np.append(Cl, Cl_test)
    
numTotClasses = np.unique(totCl).size
le = preprocessing.LabelEncoder()
totCl2 = le.fit_transform(totCl)
Cl2 = le.transform(Cl)
Cl2_test = le.transform(Cl_test)

totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(Cl).size+1)
Cl2_test = keras.utils.to_categorical(Cl2_test, num_classes=np.unique(Cl).size+1)

model = Sequential()
model.add(Dense(int(A.shape[1]/2), activation = 'relu', input_dim=A.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(int(A.shape[1]/4), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(np.unique(Cl).size+1, activation = 'softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy'])

tbLog = TensorBoard(log_dir=tb_directory, histogram_freq=0, batch_size=32,
            write_graph=True, write_grads=True, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
tbLogs = [tbLog]
model.fit(A, Cl2,
    epochs=500,
    batch_size=A.shape[1],
    callbacks = tbLogs,
    verbose=2,
    validation_split=0.01)

score = model.evaluate(A_test, Cl2_test, batch_size=A.shape[1])
model.save(model_name)
plot_model(model, to_file=model_directory+'/model.png', show_shapes=True)

print('\n  ==================================')
print('  \033[1mKeras\033[0m - Accuracy')
print('  ==================================')
print("\n  Accuracy: {:.2f}%".format(100*score[1]))
print("  Loss: {:.2f}".format(score[0]))
print('  ==================================\n')

