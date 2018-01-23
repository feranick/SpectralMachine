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
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling1D
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


df = pd.DataFrame()
for i in range(A.shape[0]):
    G = np.dstack([np.ones(En.shape[0]), En])
    G = np.dstack([G, A[0]])
    S = pd.Series(G.tolist())
    df[i] = S.values

#print(np.asarray([df.iloc[:,1].values[0]]).shape)

S = np.asarray([df.iloc[:,1].values[0]])

'''
model = Sequential()
model.add(Conv2D(1, (50, 1), input_shape=S.shape))
'''

total_time = time.clock() - start_time
print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

