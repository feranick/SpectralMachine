#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
* SpectraKeras - CNN
* 20180926a
* Uses: Keras, TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, time, pydot, graphviz, pickle, h5py
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
    
#import keras   # pure keras
import tensorflow.keras as keras  #tf.keras
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

#************************************
''' Parameters '''
#************************************
class dP:
    l_rate = 0.001
    l_rdecay = 1e-4

    CL=[30]
    CL_size=[20]
    max_pooling=100
    
    HL=[10,20,30]
    
    drop = 0.4
    l2 = 1e-4

    epochs = 100
    cv_split = 0.01
    
    #batch_size = A.shape[0]
    batch_size = 512

    plotWeightsFlag = False

#************************************
''' Parameters '''
#************************************
def main():
    start_time = time.clock()
    learnFile = sys.argv[1]

    En, A, Cl = readLearnFile(learnFile)
    learnFileRoot = os.path.splitext(learnFile)[0]

    tb_directory = "keras_CNN"
    model_directory = "."
    model_name = model_directory+"/keras_CNN_model.hd5"
    model_le = model_directory+"/keras_le.pkl"

    #totA = np.vstack((A, A_test))
    #totCl = np.append(Cl, Cl_test)
    totA = A
    totCl = Cl
    
    numTotClasses = np.unique(totCl).size
    le = preprocessing.LabelEncoder()
    totCl2 = le.fit_transform(totCl)
    Cl2 = le.transform(Cl)
    
    print(" Total number of points per data:",En.size)
    print(" Total number of classes:",numTotClasses)
    #Cl2_test = le.transform(Cl_test)
    print("\n Label Encoder saved in:", model_le,"\n")
    with open(model_le, 'ab') as f:
        f.write(pickle.dumps(le))

    totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
    Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(Cl).size+1)
    #Cl2_test = keras.utils.to_categorical(Cl2_test, num_classes=np.unique(Cl).size+1)

    # Format spectra as images for loading
    listmatrix = []
    for i in range(A.shape[0]):
        spectra = np.dstack([np.ones(En.shape[0]), En])
        spectra = np.dstack([spectra, A[i]])
        listmatrix.append(spectra)

    x_train = np.stack(listmatrix, axis=0)
    y_train = Cl2


    ### Build model
    model = keras.models.Sequential()

    for i in range(len(dP.CL)):
        model.add(keras.layers.Conv2D(dP.CL[i], (1, dP.CL_size[i]),
            activation='relu',
            input_shape=spectra.shape))

    model.add(keras.layers.MaxPooling2D(pool_size=(1, dP.max_pooling)))

    model.add(keras.layers.Dropout(dP.drop))
    model.add(keras.layers.Flatten())

    for i in range(len(dP.HL)):
        model.add(keras.layers.Dense(dP.HL[i],
            activation = 'relu',
            input_dim=A.shape[1],
            kernel_regularizer=keras.regularizers.l2(dP.l2)))
        model.add(keras.layers.Dropout(dP.drop))
    model.add(keras.layers.Dense(np.unique(Cl).size+1, activation = 'softmax'))

    #optim = opt.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    optim = keras.optimizers.Adam(lr=dP.l_rate, beta_1=0.9,
                    beta_2=0.999, epsilon=1e-08,
                    decay=dP.l_rdecay,
                    amsgrad=False)

    model.compile(loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy'])

    tbLog = keras.callbacks.TensorBoard(log_dir=tb_directory, histogram_freq=0, batch_size=dP.batch_size,
            write_graph=True, write_grads=True, write_images=True,)
    tbLogs = [tbLog]
    log = model.fit(x_train, y_train,
        epochs=dP.epochs,
        batch_size=dP.batch_size,
        callbacks = tbLogs,
        verbose=2,
        validation_split=dP.cv_split)

    accuracy = np.asarray(log.history['acc'])
    loss = np.asarray(log.history['loss'])
    val_loss = np.asarray(log.history['val_loss'])
    val_acc = np.asarray(log.history['val_acc'])

    #score = model.evaluate(A_test, Cl2_test, batch_size=A.shape[1])
    model.save(model_name)
    keras.utils.plot_model(model, to_file=model_directory+'/keras_MLP_model.png', show_shapes=True)
    
    print('\n  =============================================')
    print('  \033[1mKeras CNN\033[0m - Model Configuration')
    print('  =============================================')
    print("\n Training set file:",learnFile)
    print("\n Data size:", A.shape,"\n")
    for conf in model.get_config():
        print(conf,"\n")

    printParam()

    print('\n  ==========================================')
    print('  \033[1mKeras MLP\033[0m - Training Summary')
    print('  ==========================================')
    print("\n  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(accuracy), 100*np.amax(accuracy)))
    print("  Loss - Average: {0:.4f}; Min: {1:.4f}".format(np.average(loss), np.amin(loss)))
    print('\n\n  ==========================================')
    print('  \033[1mKeras MLP\033[0m - Validation Summary')
    print('  ==========================================')
    print("\n  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(val_acc), 100*np.amax(val_acc)))
    print("  Loss - Average: {0:.4f}; Min: {1:.4f}\n".format(np.average(val_loss), np.amin(val_loss)))
    #print("\n  Validation - Loss: {0:.2f}; accuracy: {1:.2f}%".format(score[0], 100*score[1]))
    print('  =========================================\n')

    if dP.plotWeightsFlag == True:
        plotWeights(En, A, model)

    total_time = time.clock() - start_time
    print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

#************************************
''' Open Learning Data '''
#************************************
def readLearnFile(learnFile):
    print(" Opening learning file: "+learnFile+"\n")
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        elif os.path.splitext(learnFile)[1] == ".h5":
            with h5py.File(learnFile, 'r') as hf:
                M = hf["M"][:]
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
    except:
        print("\033[1m" + " Learning file not found \n" + "\033[0m")
        return

    En = M[0,1:]
    A = M[1:,1:]
    Cl = M[1:,0]
    return En, A, Cl

#************************************
''' Print NN Info '''
#************************************
def printParam():
    print('\n  ================================================')
    print('  \033[1mKeras CNN\033[0m - Parameters')
    print('  ================================================')
    print('  Optimizer:','Adam',
                '\n  Hidden layers:', dP.HL,
                '\n  Activation function:','relu',
                '\n  L2:',dP.l2,
                '\n  Dropout:', dP.drop,
                '\n  Learning rate:', dP.l_rate,
                '\n  Learning decay rate:', dP.l_rdecay)
    #if kerasDef.fullBatch == True:
    #    print('  Full batch size: {0:d} spectra, {1:.3f} Mb'.format(A.shape[0],(1e-6*A.size*A.itemsize)))
    #else:
    print('  Batch size:', dP.batch_size)
    #print('  ================================================\n')

#************************************
''' Open Learning Data '''
#************************************
def plotWeights(En, A, model):
    import matplotlib.pyplot as plt
    plt.figure(tight_layout=True)
    plotInd = 511
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

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
