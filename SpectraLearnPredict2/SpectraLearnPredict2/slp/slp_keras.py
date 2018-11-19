#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict2 -  Keras
* Perform Machine Learning on Spectroscopy Data.
*
* Uses: Deep Neural Networks, TensorFlow, SVM, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''

import matplotlib
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Agg')

import numpy as np
import sys, os.path, getopt, glob, csv, pydot, graphviz
import random, time, configparser, os, pickle
from os.path import exists, splitext
from os import rename
from datetime import datetime, date

from .slp_config import *

#**********************************************
''' Format input data for Estimator '''
#**********************************************
def input_fn(A, Cl2):
    import tensorflow as tf
    x = tf.constant(A.astype(np.float32))
    y = tf.constant(Cl2)
    return x,y

#********************************************************************************
''' Keras '''
''' https://keras.io/getting-started/sequential-model-guide/#examples'''
#********************************************************************************
def trainKeras(En, A, Cl, A_test, Cl_test, Root):
    import tensorflow as tf
    
    # Use this to restrict GPU memory allocation in TF
    opts = tf.GPUOptions(per_process_gpu_memory_fraction=sysDef.fractionGPUmemory)
    conf = tf.ConfigProto(gpu_options=opts)
    #conf.gpu_options.allow_growth = True
    
    if kerasDef.useTFKeras:
        import tensorflow.keras as keras  #tf.keras
        tf.Session(config=conf)
    else:
        import keras   # pure keras
        from keras.backend.tensorflow_backend import set_session
        set_session(tf.Session(config=conf))
    
    from sklearn import preprocessing
    from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
    
    tb_directory = "keras_" + str(len(kerasDef.hidden_layers))+"HL_"+str(kerasDef.hidden_layers[0])
    model_directory = "."
    if kerasDef.regressor:
        model_name = model_directory+"/keras_regressor_"+str(len(kerasDef.hidden_layers))+"HL_"+str(kerasDef.hidden_layers[0])+".hd5"
    else:
        model_name = model_directory+"/keras_"+str(len(kerasDef.hidden_layers))+"HL_"+str(kerasDef.hidden_layers[0])+".hd5"
    model_le = model_directory+"/keras_model_le.pkl"
    
    if kerasDef.alwaysRetrain == False:
        print(" Training model saved in: ", model_name, "\n")
    else:
        kerasDef.alwaysImprove = False
        print(" Training model not saved\n")
    
    #**********************************************
    ''' Initialize Estimator and training data '''
    #**********************************************
    print(' Preprocessing data and classes for Keras\n')

    totA = np.vstack((A, A_test))
    totCl = np.append(Cl, Cl_test)

    
    if kerasDef.regressor:
        Cl2 = np.copy(Cl)
        Cl2_test = np.copy(Cl_test)
        le = None
    else:

        numTotClasses = np.unique(totCl).size
        le = preprocessing.LabelEncoder()
        totCl2 = le.fit_transform(totCl)
        Cl2 = le.transform(Cl)
        Cl2_test = le.transform(Cl_test)

        totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
        Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(totCl).size+1)
        Cl2_test = keras.utils.to_categorical(Cl2_test, num_classes=np.unique(totCl).size+1)
        print(" Label Encoder saved in:", model_le)
        with open(model_le, 'ab') as f:
            f.write(pickle.dumps(le))
    
    if kerasDef.fullBatch == True:
        batch_size = A.shape[0]
    else:
        batch_size = kerasDef.batchSize

    printParamKeras(A)

    if kerasDef.alwaysImprove == True or os.path.exists(model_name) is False:
        model = keras.models.Sequential()
        for numLayers in kerasDef.hidden_layers:
            model.add(keras.layers.Dense(numLayers,
                    activation = kerasDef.activation_function,
                    input_dim=A.shape[1],
                    kernel_regularizer=keras.regularizers.l2(kerasDef.l2_reg_strength)))
            model.add(keras.layers.Dropout(kerasDef.dropout_perc))

        if kerasDef.regressor:
            model.add(keras.layers.Dense(1))
            model.compile(loss='mse',
                optimizer=kerasDef.optimizer,
                metrics=['mae'])
        else:
            model.add(keras.layers.Dense(np.unique(totCl).size+1, activation = 'softmax'))
            model.compile(loss='categorical_crossentropy',
              optimizer=kerasDef.optimizer,
              metrics=['accuracy'])

        tbLog = keras.callbacks.TensorBoard(log_dir=tb_directory, histogram_freq=kerasDef.tbHistogramFreq,
                batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)
        #tbLog.set_model(model)
        tbLogs = [tbLog]
        log = model.fit(A, Cl2,
            epochs=kerasDef.trainingSteps,
            batch_size=batch_size,
            callbacks = tbLogs,
            verbose = 2,
            validation_data=(A_test, Cl2_test))

        loss = np.asarray(log.history['loss'])
        val_loss = np.asarray(log.history['val_loss'])

        if kerasDef.regressor:
            accuracy = None
            val_acc = None
        else:
            accuracy = np.asarray(log.history['acc'])
            val_acc = np.asarray(log.history['val_acc'])

        model.save(model_name)

        if kerasDef.plotModel == True:
            from keras.utils import plot_model
            keras.utils.plot_model(model, to_file=model_directory+'/keras_MLP_model.png', show_shapes=True)
            
            import matplotlib.pyplot as plt
            plt.figure(tight_layout=True)
            plotInd = int(len(kerasDef.hidden_layers))*100+11
            visibleX = True
            for layer in model.layers:
                try:
                    w_layer = layer.get_weights()[0]
                    ax = plt.subplot(plotInd)
                    newX = np.arange(En[0], En[-1], (En[-1]-En[0])/w_layer.shape[0])
                    plt.plot(En, np.interp(En, newX, w_layer[:,0]), label=layer.get_config()['name'])
                    plt.legend(loc='upper right')
                    plt.setp(ax.get_xticklabels(), visible=visibleX)
                    visibleX = False
                    plotInd +=1
                except:
                    pass

            plt.xlabel('Raman shift [1/cm]')
            plt.legend(loc='upper right')
            plt.savefig('keras_weights_MLP' + '.png', dpi = 160, format = 'png')  # Save plot

        printModelKeras(model)

        print("\n  Number of spectra = ",A.shape[0])
        print("  Number of points in each spectra = ", A.shape[1])
        if kerasDef.regressor == False:
            print("  Number unique classes (training): ", np.unique(Cl).size)
            print("  Number unique classes (validation):", np.unique(Cl_test).size)
            print("  Number unique classes (total): ", np.unique(totCl).size)

        printParamKeras(A)
        printTrainSummary(accuracy, loss, val_acc, val_loss)
        
    else:
        print(" Retreaving training model from: ", model_name,"\n")
        model = keras.models.load_model(model_name)
        printModelKeras(model)
        printParamKeras(A)
    
    score = model.evaluate(A_test, Cl2_test, batch_size=batch_size)
    printEvalSummary(model_name, score)
    return model, le

#***********************************************
''' Summary visualization panels '''
#***********************************************
def printTrainSummary(accuracy,loss, val_acc, val_loss):
    if kerasDef.regressor:
        print('\n  ========================================================')
        print('  \033[1mKeras MLP - Regressor\033[0m - Training Summary')
        print('  ========================================================')
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(loss), np.amin(loss), loss[-1]))
        print('  ========================================================\n')
    else:
        print('\n  ================================================')
        print('  \033[1mKeras MLP - Classifier \033[0m - Training Summary')
        print('  ================================================')
        print("  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(accuracy), 100*np.amax(accuracy)))
        print("  Loss - Average: {0:.4f}; Min: {1:.4f}".format(np.average(loss), np.amin(loss)))
        print("\n  Validation ({0:.0f}%) - Average: {1:.2f}%; Max: {2:.2f}%".format(100*preprocDef.percentCrossValid,
                100*np.average(val_acc), 100*np.amax(val_acc)))
        print("  Loss - Average: {0:.4f}; Min: {1:.4f}".format(np.average(val_loss), np.amin(val_loss)))
        print('  ================================================\n\n')

def printEvalSummary(model_name, score):
    print('\n\n  ================================================')
    print('  \033[1mKeras MLP\033[0m - Evaluation Summary')
    print('  ================================================')
    print("  Evaluation ({0:.0f}%) - Loss: {1:.4f}".format(100*preprocDef.percentCrossValid,
                score[0]))
    if kerasDef.regressor == False:
        print("  Accuracy: {0:.2f}%".format(100*score[1]))
    print("  Global step: {:d}".format(kerasDef.trainingSteps))
    if os.path.exists(model_name) is True:
        print("\n  Model saved in:", model_name)
    print('  ================================================\n')

def printModelKeras(model):
    print('\n  ================================================')
    print('  \033[1mKeras MLP\033[0m - Model Configuration')
    print('  ================================================')
    for conf in model.get_config():
        print(conf,"\n")
    model.summary()

def printParamKeras(A):
    print('\n  ================================================')
    if kerasDef.regressor:
        print('  \033[1mKeras MLP - Regressor \033[0m - Parameters')
    else:
        print('  \033[1mKeras MLP - Classifier \033[0m - Parameters')
    print('  ================================================')
    print('  Optimizer:',kerasDef.optimizer_tag,
                '\n  Hidden layers:', kerasDef.hidden_layers,
                '\n  Activation function:',kerasDef.activation_function,
                '\n  L2:',kerasDef.l2_reg_strength,
                '\n  Dropout:', kerasDef.dropout_perc,
                '\n  Learning rate:', kerasDef.learning_rate,
                '\n  Learning decay rate:', kerasDef.learning_decay_rate)
    if kerasDef.fullBatch == True:
        print('  Full batch size: {0:d} spectra, {1:.3f} Mb'.format(A.shape[0],(1e-6*A.size*A.itemsize)))
    else:
        print('  Batch size:', kerasDef.batchSize)
    print('  ================================================\n')

#********************************************************************************
''' Predict using Keras model '''
#********************************************************************************
def predKeras(model, le, R, Cl):
    if kerasDef.useTFKeras:
        import tensorflow.keras as keras  #tf.keras
    else:
        import keras   # pure keras
    
    from sklearn import preprocessing
    
    if kerasDef.regressor:
        predictions = model.predict(R).flatten()[0]
        print('  ========================================================')
        print('  \033[1mKeras MLP - Regressor\033[0m - Prediction')
        print('  ========================================================')
        predValue = predictions
        print('\033[1m\n  Predicted value = {0:.4f}\033[0m\n'.format(predValue))
        print('  ========================================================\n')
        predProb = 0
    else:
        predictions = model.predict(R, verbose=1)
        pred_class = np.argmax(predictions)
        if pred_class.size >0:
            predValue = le.inverse_transform([pred_class])[0]
        else:
            predValue = 0

        predProb = round(100*predictions[0][pred_class],2)
        print('\n  ==================================')
        print('  \033[1mKeras\033[0m - Probability >',str(kerasDef.thresholdProbabilityPred),'%')
        print('  ==================================')
        print('  Prediction\tProbability [%]')
        for i in range(len(predictions[0])-1):
            if predictions[0][i]>kerasDef.thresholdProbabilityPred:
                print(' ',str(np.unique(Cl)[i]),'\t\t',
                    str('{:.2f}'.format(100*predictions[0][i])))
        print('  ==================================')
    
        print('\033[1m' + '\n Predicted value (Keras) = ' + str(predValue) +
            '  (probability = ' + str(predProb) + '%)\033[0m\n')

    return predValue, predProb



