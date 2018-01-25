#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraKeras - MLP
*
* 20180124a
*
* Uses: Keras, TensorFlow
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
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

#A, A_test, Cl, Cl_test = train_test_split(A, Cl, test_size=0.01, random_state=42)
#En_test = En

tb_directory = "keras_MLP"
model_directory = "."
model_name = model_directory+"/keras_model_MLP.hd5"

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

batch_size = A.shape[1]
#batch_size = 64

plt.plot(En, A[0], label='Training data')
plt.savefig(learnFileRoot + '.png', dpi = 160, format = 'png')  # Save plot
plt.close()
#plt.show()

model = Sequential()
model.add(Dense(200, activation = 'relu', input_dim=A.shape[1],
    kernel_regularizer=regularizers.l2(1e-4),
    name='dense1'))
model.add(Dropout(0.3,name='drop2'))
model.add(Dense(200, activation = 'relu',
    kernel_regularizer=regularizers.l2(1e-4),
    name='dense3'))
model.add(Dropout(0.3,
    name='drop4'))
model.add(Dense(np.unique(Cl).size+1, activation = 'softmax',
    name='dense5'))

#optim = opt.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
optim = opt.Adam(lr=0.0001, beta_1=0.9,
                                        beta_2=0.999, epsilon=1e-08,
                                        decay=1e-7,
                                        amsgrad=False)

model.compile(loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'])

tbLog = TensorBoard(log_dir=tb_directory, histogram_freq=0, batch_size=batch_size,
            write_graph=True, write_grads=True, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
tbLogs = [tbLog]
log = model.fit(A, Cl2,
    epochs=200,
    batch_size=batch_size,
    callbacks = tbLogs,
    verbose=2,
    validation_split=0.05)

accuracy = np.asarray(log.history['acc'])
loss = np.asarray(log.history['loss'])
val_loss = np.asarray(log.history['val_loss'])
val_acc = np.asarray(log.history['val_acc'])


#score = model.evaluate(A_test, Cl2_test, batch_size=A.shape[1])
model.save(model_name)
plot_model(model, to_file=model_directory+'/model.png', show_shapes=True)

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



for layer in model.layers:
    try:
        print(layer.get_config()['name'])
        w_layer = layer.get_weights()[0]
        newX = np.arange(En[0], En[-1], (En[-1]-En[0])/w_layer.shape[0])
        plt.plot(En, np.interp(En, newX, w_layer[:,0]), label=layer.get_config()['name'])
        plt.savefig(layer.get_config()['name'] + '.png', dpi = 160, format = 'png')  # Save plot
        plt.close()
    except:
        pass

#plt.xlabel('Raman shift [1/cm]')
#plt.legend(loc='upper right')
#plt.show()

#w_layer1 = model.get_layer('dense1').get_weights()[0]
#plt.plot(En, w_layer1[:,0], label='convoluted data')
#plt.show()

total_time = time.clock() - start_time
print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

