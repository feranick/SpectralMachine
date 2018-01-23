#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraKeras
*
* 20180122a
*
* Uses: Deep Neural Networks, TensorFlow, SVM, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
import numpy as np
import keras, sys, os.path, time
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
import keras.optimizers as opt
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
import tensorflow as tf
import pandas as pd

start_time = time.clock()
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
totA = A
totCl = Cl
numTotClasses = np.unique(totCl).size
le = preprocessing.LabelEncoder()
totCl2 = le.fit_transform(totCl)
Cl2 = le.transform(Cl)
totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(Cl).size+1)



df = pd.DataFrame()
for i in range(A.shape[0]):
    G = np.dstack([np.ones(En.shape[0]), En])
    G = np.dstack([G, A[0]])
    S = pd.Series(G.tolist())
    df[i] = S.values


S = np.asarray([df.iloc[:,1].values[0]])

print(S.shape)

#x_train = df.tolist()
#y_train = Cl2

model = Sequential()
model.add(Conv2D(3, (1, 50), activation='relu', input_shape=S.shape))
model.add(Conv2D(3, (1, 50), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 10)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
optim = opt.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optim)


#model.fit(x_train, y_train, batch_size=32, epochs=10)
#score = model.evaluate(x_test, y_test, batch_size=32)


total_time = time.clock() - start_time
print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

