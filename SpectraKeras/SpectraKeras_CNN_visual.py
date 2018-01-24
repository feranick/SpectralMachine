#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraKeras - CNN
*
* 20180123a
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
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
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

# Prepare data
En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
M = np.delete(M,np.s_[0:1],0)
Cl = ['{:.2f}'.format(x) for x in M[:,0]]
A = np.delete(M,np.s_[0:1],1)
learnFileRoot = os.path.splitext(learnFile)[0]

# Format Classes
totA = A
totCl = Cl
numTotClasses = np.unique(totCl).size
le = preprocessing.LabelEncoder()
totCl2 = le.fit_transform(totCl)
Cl2 = le.transform(Cl)
totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(Cl).size+1)

batch_size = A.shape[1]
#batch_size = 64

tb_directory = "keras_CNN"
model_directory = "."
model_name = model_directory+"/keras_model_CNN.hd5"

# Format spectra as images for loading
listmatrix = []
for i in range(A.shape[0]):
    spectra = np.dstack([np.ones(En.shape[0]), En])
    spectra = np.dstack([spectra, A[i]])
    listmatrix.append(spectra)

#print(spectra[:,:,2])
x_train = np.stack(listmatrix, axis=0)
y_train = Cl2

plt.plot(En, A[0], label='Training data')
#plt.show()

model1 = Sequential()
layer = Conv2D(3, (1, 200), activation = 'relu', input_shape=spectra.shape,
    name='conv1')
model1.add(layer)
weigths = layer.get_weights()
conv = model1.predict(x_train)
test = conv[1]
print(conv.shape)
plt.plot(range(0, test.shape[1]), test[:,:,2][0], label='convoluted data')

#print("Weigths: ", weigths[0].shape)
#print(weigths[0][:,:,2][0])
#plt.plot(range(0, weigths[0].shape[1]), weigths[0][:,:,2][0], label='convoluted data')

plt.show()

########################
## Actual Training
########################

# Setup Model
model = Sequential()
model.add(Conv2D(3, (1, 20), activation='relu',
    name='conv1',
    input_shape=spectra.shape))
model.add(Conv2D(3, (1, 20), activation='relu',
    name='conv2'))
model.add(MaxPooling2D(pool_size=(1, 20),
    name='maxpool3'))
model.add(Dropout(0.25, name='drop4'))
model.add(Flatten(name='flat5'))
model.add(Dense(256, activation='relu',
    name='dense6'))
model.add(Dropout(0.5, name='drop7'))
model.add(Dense(np.unique(Cl).size+1, activation='softmax',
    name='dense8'))
optim = opt.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

tbLog = TensorBoard(log_dir=tb_directory, histogram_freq=0, batch_size=batch_size,
            write_graph=True, write_grads=True, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
tbLogs = [tbLog]

log = model.fit(x_train, y_train,
    batch_size=32,
    epochs=10,
    callbacks = tbLogs,
    verbose=2,
    validation_split=0.05)
#score = model.evaluate(x_test, y_test, batch_size=32)

accuracy = np.asarray(log.history['acc'])
loss = np.asarray(log.history['loss'])
val_loss = np.asarray(log.history['val_loss'])
val_acc = np.asarray(log.history['val_acc'])

model.save(model_name)
plot_model(model, to_file=model_directory+'/model.png', show_shapes=True)

print('\n  ==========================================')
print('  \033[1mKeras CNN\033[0m - Training Summary')
print('  ==========================================')
print("\n  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(accuracy), 100*np.amax(accuracy)))
print("  Loss - Average: {0:.4f}; Min: {1:.4f}".format(np.average(loss), np.amin(loss)))
print('\n\n  ==========================================')
print('  \033[1mKeras CNN\033[0m - Validation Summary')
print('  ==========================================')
print("\n  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(val_acc), 100*np.amax(val_acc)))
print("  Loss - Average: {0:.4f}; Min: {1:.4f}\n".format(np.average(val_loss), np.amin(val_loss)))
#print("\n  Validation - Loss: {0:.2f}; accuracy: {1:.2f}%".format(score[0], 100*score[1]))
print('  =========================================\n')


total_time = time.clock() - start_time
print("\n Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

