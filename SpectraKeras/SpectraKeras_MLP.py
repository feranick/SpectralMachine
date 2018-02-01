#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
* SpectraKeras - MLP
* 20180129a
* Uses: Keras, TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)

import numpy as np
import keras, sys, os.path, time
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, ActivityRegularization, MaxPooling1D
import keras.optimizers as opt
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
import tensorflow as tf
import matplotlib.pyplot as plt

start_time = time.clock()
learnFile = sys.argv[1]
print("\n Training set file:",learnFile,"\n")

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

##########################
# Parameters
##########################

l_rate = 1e-4
l_rdecay = 1e-7

HL1 = 1000
drop1 = 0.5
l2_1 = 1e-3
HL2 = 1000
drop2 = 0.5
l2_2 = 1e-3
epochs = 100
cv_split = 0.05

batch_size = A.shape[0]
#batch_size = 64
#########################

tb_directory = "keras_MLP"
model_directory = "."
model_name = model_directory+"/keras_MLP_model.hd5"

#totA = np.vstack((A, A_test))
#totCl = np.append(Cl, Cl_test)
totA = A
totCl = Cl
    
numTotClasses = np.unique(totCl).size
le = preprocessing.LabelEncoder()
totCl2 = le.fit_transform(totCl)
Cl2 = le.transform(Cl)
#Cl2_test = le.transform(Cl_test)

totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(Cl).size+1)
#Cl2_test = keras.utils.to_categorical(Cl2_test, num_classes=np.unique(Cl).size+1)

### Build model
model = Sequential()
model.add(Dense(HL1, activation = 'relu', input_dim=A.shape[1],
    kernel_regularizer=regularizers.l2(l2_1),
    name='dense1'))
model.add(Dropout(drop1,name='drop1'))
model.add(Dense(HL2, activation = 'relu',
    kernel_regularizer=regularizers.l2(12_2),
    name='dense2'))
model.add(Dropout(drop2,
    name='drop2'))
model.add(Dense(np.unique(Cl).size+1, activation = 'softmax',
    name='dense3'))

#optim = opt.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
optim = opt.Adam(lr=l_rate, beta_1=0.9,
                    beta_2=0.999, epsilon=1e-08,
                    decay=l_rdecay,
                    amsgrad=False)

model.compile(loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'])

tbLog = TensorBoard(log_dir=tb_directory, histogram_freq=0, batch_size=batch_size,
            write_graph=True, write_grads=True, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
tbLogs = [tbLog]
log = model.fit(A, Cl2,
    epochs=epochs,
    batch_size=batch_size,
    callbacks = tbLogs,
    verbose=2,
    validation_split=cv_split)

accuracy = np.asarray(log.history['acc'])
loss = np.asarray(log.history['loss'])
val_loss = np.asarray(log.history['val_loss'])
val_acc = np.asarray(log.history['val_acc'])

#score = model.evaluate(A_test, Cl2_test, batch_size=A.shape[1])
model.save(model_name)
plot_model(model, to_file=model_directory+'/keras_MLP_model.png', show_shapes=True)
print('\n  =============================================')
print('  \033[1mKeras MLP\033[0m - Model Configuration')
print('  =============================================')
print("\n Training set file:",learnFile)
print("\n Data size:", A.shape,"\n")
for conf in model.get_config():
    print(conf,"\n")

print('\n  ==========================================')
print('  \033[1mKeras MLP\033[0m - Training Summary')
print('  ==========================================')
print("\n  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(accuracy), 100*np.amax(accuracy)))
print("  Loss - Average: {0:.4f}; Min: {1:.4f}".format(np.average(loss), np.amin(loss)))
print('\n\n  ==========================================')
print('  \033[1mKerasMLP\033[0m - Validation Summary')
print('  ==========================================')
print("\n  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(val_acc), 100*np.amax(val_acc)))
print("  Loss - Average: {0:.4f}; Min: {1:.4f}\n".format(np.average(val_loss), np.amin(val_loss)))
#print("\n  Validation - Loss: {0:.2f}; accuracy: {1:.2f}%".format(score[0], 100*score[1]))
print('  =========================================\n')


### Plotting weights
plt.figure(tight_layout=True)
plotInd = 411
for layer in model.layers:
    try:
        w_layer = layer.get_weights()[0]
        ax = plt.subplot(plotInd)
        newX = np.arange(En[0], En[-1], (En[-1]-En[0])/w_layer.shape[0])
        plt.plot(En, np.interp(En, newX, w_layer[:,0]), label=layer.get_config()['name'])
        plt.legend(loc='upper right')
        plt.setp(ax.get_xticklabels(), visible=False)
        plotInd +=1
    except:
        pass

ax1 = plt.subplot(plotInd)
ax1.plot(En, A[0], label='Sample data')

plt.xlabel('Raman shift [1/cm]')
plt.legend(loc='upper right')
plt.savefig('keras_MLP_weights' + '.png', dpi = 160, format = 'png')  # Save plot

####################

total_time = time.clock() - start_time
print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

