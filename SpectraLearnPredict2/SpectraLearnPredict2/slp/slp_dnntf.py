#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict2 -  DNN-TF
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
import sys, os.path, getopt, glob, csv, pickle
import random, time, configparser, os
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
''' TensorFlow '''
''' Run tf.estimator.DNNClassifier or tf.estimator.DNNRegressor'''
''' https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier'''
''' https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor'''
#********************************************************************************
''' Train DNNClassifier model training via TensorFlow-Estimators '''
#********************************************************************************
def trainDNNTF(A, Cl, A_test, Cl_test, Root):
    import tensorflow as tf
    #import tensorflow.contrib.learn as skflow
    #from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
    
    if dnntfDef.logCheckpoint == True:
        tf.logging.set_verbosity(tf.logging.INFO)
    
    if dnntfDef.alwaysRetrain == False:
        model_directory = Root + "/DNN-TF_" + str(len(dnntfDef.hidden_layers))+"HL_"+str(dnntfDef.hidden_layers[0])
        print("\n  Training model saved in: ", model_directory, "\n")
    else:
        dnntfDef.alwaysImprove = True
        model_directory = None
        print("\n  Training model not saved\n")

    #**********************************************
    ''' Initialize Estimator and training data '''
    #**********************************************
    print(' Initializing TensorFlow...')
    tf.reset_default_graph()

    totA = np.vstack((A, A_test))
    totCl = np.append(Cl, Cl_test)
    numTotClasses = np.unique(totCl).size
    
    ###############################
    ### Uncomment for DNNClassifier
    if dnntfDef.useRegressor == False:
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        totCl2 = le.fit_transform(totCl)
        Cl2 = le.transform(Cl)
        Cl2_test = le.transform(Cl_test)
        model_le = "dnntf_le.pkl"
        print("\n Label Encoder saved in:", model_le)
        with open(model_le, 'ab') as f:
            f.write(pickle.dumps(le))
    else:
        le = 0
        totCl2 = totCl
        Cl2 = Cl
        Cl2_test = Cl_test
    #############################
    
    if dnntfDef.fullBatch == True:
        batch_size_train = A.shape[0]
        batch_size_test = A_test.shape[0]
    else:
        batch_size_train = dnntfDef.batchSize
        batch_size_test = dnntfDef.batchSize

    printInfo(A)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": A},
            y = Cl2,
            num_epochs=None,
            batch_size=batch_size_train,            # Default: 128
            queue_capacity=dnntfDef.queueCapacity,  # Default: 1000
            shuffle=dnntfDef.shuffleTrain,
            num_threads=dnntfDef.numThreadsInput)
                    
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": A_test},
            y = Cl2_test,
            num_epochs=1,
            batch_size=batch_size_test,   # Default: 128
            queue_capacity=1000,          # Default: 1000
            shuffle=dnntfDef.shuffleTest,
            num_threads=1)
    
    feature_columns = [tf.feature_column.numeric_column("x", shape=[totA.shape[1]])]
    
    print(tf.feature_column.numeric_column("x", shape=[totA.shape[1]]))
    
    #**********************************************
    ''' Define learning rate '''
    #**********************************************
    if dnntfDef.learning_rate_decay == False:
        learning_rate = dnntfDef.learning_rate
    else:
        learning_rate = tf.train.exponential_decay(dnntfDef.learning_rate,
                        tf.Variable(0, trainable=False),
                        dnntfDef.learning_rate_decay_steps,
                        dnntfDef.learning_rate_decay_rate,
                        staircase=True)

    # Use this to restrict GPU memory allocation in TF
    opts = tf.GPUOptions(per_process_gpu_memory_fraction=sysDef.fractionGPUmemory)
    conf = tf.ConfigProto(gpu_options=opts)
    #conf.gpu_options.allow_growth = True
    
    ###############################
    if dnntfDef.useRegressor == False:
        clf = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=dnntfDef.hidden_layers,
            optimizer=dnntfDef.optimizer, n_classes=numTotClasses,
            activation_fn=dnntfDef.activationFn, model_dir=model_directory,
            config=tf.estimator.RunConfig().replace(session_config=conf, save_checkpoints_secs=dnntfDef.timeCheckpoint),
            dropout=dnntfDef.dropout_perc)
    else:
        clf = tf.estimator.DNNRegressor(feature_columns=feature_columns, hidden_units=dnntfDef.hidden_layers,
            optimizer=dnntfDef.optimizer,
            activation_fn=dnntfDef.activationFn, model_dir=model_directory,
            config=tf.estimator.RunConfig().replace(session_config=conf, save_checkpoints_secs=dnntfDef.timeCheckpoint),
            dropout=dnntfDef.dropout_perc)
    ###############################

    '''
    # Validation monitors are deprecated
    validation_monitor = [skflow.monitors.ValidationMonitor(input_fn=test_input_fn,
                                                           eval_steps=1,
                                                           every_n_steps=dnntfDef.valMonitorSecs)]
    hooks = monitor_lib.replace_monitors_with_hooks(validation_monitor, clf)
    '''

    hooks = [
        tf.train.SummarySaverHook(
            save_secs = dnntfDef.valMonitorSecs,
            output_dir=model_directory,
            scaffold= tf.train.Scaffold(),
            summary_op=tf.summary.merge_all()),
        tf.train.ProfilerHook(save_secs=60,
            output_dir=model_directory+"/profiler",
            show_memory = False,
            show_dataflow = False),
        ]

    #**********************************************
    ''' Define parameters for savedmodel '''
    #**********************************************
    feature_spec = {'x': tf.FixedLenFeature([numTotClasses],tf.float32)}
    def serving_input_receiver_fn():
        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='input_tensors')
        receiver_tensors = {'inputs': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    print("\n Number of global steps:",dnntfDef.trainingSteps)

    #**********************************************
    ''' Train '''
    #**********************************************
    if dnntfDef.alwaysImprove == True or os.path.exists(model_directory) is False:
        print(" (Re-)training using dataset: ", Root,"\n")
        clf.train(input_fn=train_input_fn,
                steps=dnntfDef.trainingSteps, hooks=hooks)
        print(" Exporting savedmodel in: ", Root,"\n")
        clf.export_savedmodel(model_directory, serving_input_receiver_fn)
    else:
        print("  Retreaving training model from: ", model_directory,"\n")

    accuracy_score = clf.evaluate(input_fn=test_input_fn, steps=1, hooks=hooks)
    printInfo(A)

    print('\n  ==================================')
    if dnntfDef.useRegressor == False:
        print('  \033[1mtf.DNN Classifier \033[0m - Accuracy')
        print("\n  Accuracy: {:.2f}%".format(100*accuracy_score["accuracy"])) # this is for classifier
    else:
        print('  \033[1mtf.DNN Regressor \033[0m - Prediction')
    print('  ==================================')
    print("  Average Loss: {:.3f}".format(accuracy_score["average_loss"]))
    print("  Loss: {:.2f}".format(accuracy_score["loss"]))
    print("  Global step: {:.2f}\n".format(accuracy_score["global_step"]))
    print('  ==================================\n')

    return clf, le

def printInfo(A):
    print('==========================================================================\n')
    if dnntfDef.useRegressor == False:
        print('\033[1m Running Deep Neural Networks: tf.DNN Classifier - TensorFlow...\033[0m')
    else:
        print('\033[1m Running Deep Neural Networks: tf.DNN Regressor - TensorFlow...\033[0m')
    print('  Optimizer:',dnntfDef.optimizer_tag,
                '\n  Hidden layers:', dnntfDef.hidden_layers,
                '\n  Activation function:',dnntfDef.activation_function,
                '\n  L2:',dnntfDef.l2_reg_strength,
                '\n  Dropout:', dnntfDef.dropout_perc,
                '\n  Learning rate:', dnntfDef.learning_rate,
                '\n  Shuffle Train:', dnntfDef.shuffleTrain,
                '\n  Shuffle Test:', dnntfDef.shuffleTest,)
    if dnntfDef.learning_rate_decay == False:
        print('  Fixed learning rate :',dnntfDef.learning_rate,)
    else:
        print('  Exponential decay - initial learning rate:',dnntfDef.learning_rate,
                '\n  Exponential decay rate:', dnntfDef.learning_rate_decay_rate,
                '\n  Exponential decay steps:', dnntfDef.learning_rate_decay_steps,)
    if dnntfDef.fullBatch == True:
        print('  Full batch size: {0:d} spectra, {1:.3f} Mb'.format(A.shape[0],(1e-6*A.size*A.itemsize)))
    else:
        print('  Batch size:', dnntfDef.batchSize)
    print('  Queue capacity:', dnntfDef.queueCapacity, '\n')

#********************************************************************************
''' Predict using tf.estimator.DNNClassifier or tf.estimator.DNNRegressor
    model via TensorFlow '''
#********************************************************************************
def predDNNTF(clf, le, R, Cl):
    import tensorflow as tf
    #import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": R},
      num_epochs=1,
      shuffle=False)
      
    predictions = list(clf.predict(input_fn=predict_input_fn))
    if dnntfDef.useRegressor == False:
        pred_class = [p["class_ids"] for p in predictions][0][0]
    
        if pred_class.size >0:
            predValue = le.inverse_transform(pred_class)
        else:
            predValue = 0
        prob = [p["probabilities"] for p in predictions][0]
        predProb = round(100*prob[pred_class],2)
        rosterPred = np.where(prob>dnntfDef.thresholdProbabilityPred/100)[0]
        print('\n  =============================================')
        print('  \033[1mtf.DNN Classifier-TF\033[0m - Probability >',str(dnntfDef.thresholdProbabilityPred),'%')
        print('  =============================================')
        print('  Prediction\tProbability [%]')
        for i in range(rosterPred.shape[0]):
            print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.4f}'.format(100*prob[rosterPred][i])))
        print('  =============================================')
    
        print('\033[1m' + '\n Predicted value (tf.DNNClassifier) = ' + str(predValue) +
            '  (probability = ' + str(predProb) + '%)\033[0m\n')
    else:
        pred = [p["predictions"] for p in predictions][0][0]
        predProb = 0
        if pred.size >0:
            predValue = pred
        else:
            predValue = 0
        print('\n  ==================================')
        print('  \033[1mtf.DNN Regressor - TF\033[0m')
        print('  ==================================')
        print('\033[1m' + '\n Predicted value (tf.DNNRegressor) = ' + str(predValue) +'\033[0m\n')

    return predValue, predProb

#********************************************************************************
''' TensorFlow '''
''' Run SkFlow - DNN Classifier '''
''' https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier'''
#********************************************************************************
''' Train DNNClassifier model training via TensorFlow-skflow '''
#********************************************************************************
def trainDNNTF2(A, Cl, A_test, Cl_test, Root):
    print('==========================================================================\n')
    print('\033[1m Running Deep Neural Networks: skflow-DNNClassifier - TensorFlow...\033[0m')
    print('  Hidden layers:', dnntfDef.hidden_layers)
    print('  Optimizer:',dnntfDef.optimizer_tag,
                '\n  Activation function:',dnntfDef.activation_function,
                '\n  L2:',dnntfDef.l2_reg_strength,
                '\n  Dropout:', dnntfDef.dropout_perc)
    import tensorflow as tf
    import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing
    
    if dnntfDef.logCheckpoint ==True:
        tf.logging.set_verbosity(tf.logging.INFO)
    
    if dnntfDef.alwaysRetrain == False:
        model_directory = Root + "/DNN-TF-SK_" + str(len(dnntfDef.hidden_layers))+"HL_"+str(dnntfDef.hidden_layers[0])
        print("\n  Training model saved in: ", model_directory, "\n")
    else:
        dnntfDef.alwaysImprove = True
        model_directory = None
        print("\n  Training model not saved\n")

    #**********************************************
    ''' Initialize Estimator and training data '''
    #**********************************************
    print(' Initializing TensorFlow...')
    tf.reset_default_graph()

    totA = np.vstack((A, A_test))
    totCl = np.append(Cl, Cl_test)
    numTotClasses = np.unique(totCl).size
    
    le = preprocessing.LabelEncoder()
    totCl2 = le.fit_transform(totCl)
    Cl2 = le.transform(Cl)
    Cl2_test = le.transform(Cl_test)
    
    validation_monitor = skflow.monitors.ValidationMonitor(input_fn=lambda: input_fn(A_test, Cl2_test),
                                                           eval_steps=1,
                                                           every_n_steps=dnntfDef.valMonitorSecs)

    feature_columns = skflow.infer_real_valued_columns_from_input(totA.astype(np.float32))
    
    '''
    clf = skflow.DNNClassifier(feature_columns=feature_columns, hidden_units=dnntfDef.hidden_layers,
            optimizer=dnntfDef.optimizer, n_classes=numTotClasses,
            activation_fn=dnntfDef.activationFn, model_dir=model_directory,
            config=skflow.RunConfig(save_checkpoints_secs=dnntfDef.timeCheckpoint),
            dropout=dnntfDef.dropout_perc)
    '''
    clf = skflow.DNNClassifier(feature_columns=feature_columns, hidden_units=dnntfDef.hidden_layers,
            optimizer=dnntfDef.optimizer, n_classes=numTotClasses,
            activation_fn=dnntfDef.activationFn, model_dir=model_directory,
            config=tf.estimator.RunConfig().replace(save_summary_steps=dnntfDef.timeCheckpoint),
            dropout=dnntfDef.dropout_perc)
    print("\n Number of global steps:",dnntfDef.trainingSteps)

    #**********************************************
    ''' Train '''
    #**********************************************
    if dnntfDef.alwaysImprove == True or os.path.exists(model_directory) is False:
        print(" (Re-)training using dataset: ", Root,"\n")
        clf.fit(input_fn=lambda: input_fn(A, Cl2),
                steps=dnntfDef.trainingSteps, monitors=[validation_monitor])
    else:
        print("  Retreaving training model from: ", model_directory,"\n")

    accuracy_score = clf.evaluate(input_fn=lambda: input_fn(A_test, Cl2_test), steps=1)
    print('\n  ===================================')
    print('  \033[1msk-DNN-TF\033[0m - Accuracy')
    print('  ===================================')
    print("\n  Accuracy: {:.2f}%".format(100*accuracy_score["accuracy"]))
    print("  Loss: {:.2f}".format(accuracy_score["loss"]))
    print("  Global step: {:.2f}\n".format(accuracy_score["global_step"]))
    print('  ===================================\n')

    return clf, le

#********************************************************************************
''' Predict using DNNClassifier model via TensorFlow-skflow '''
#********************************************************************************
def predDNNTF2(clf, le, R, Cl):
    import tensorflow as tf
    import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing

    #**********************************************
    ''' Predict '''
    #**********************************************
    def input_fn_predict():
        x = tf.constant(R.astype(np.float32))
        return x

    pred_class = list(clf.predict_classes(input_fn=input_fn_predict))[0]
    predValue = le.inverse_transform(pred_class)
    prob = list(clf.predict_proba(input_fn=input_fn_predict))[0]
    predProb = round(100*prob[pred_class],2)
    
    rosterPred = np.where(prob>dnntfDef.thresholdProbabilityPred/100)[0]
    
    print('\n  ===================================')
    print('  \033[1msk-DNN-TF\033[0m - Probability >',str(dnntfDef.thresholdProbabilityPred),'%')
    print('  ===================================')
    print('  Prediction\tProbability [%]')
    for i in range(rosterPred.shape[0]):
        print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.4f}'.format(100*prob[rosterPred][i])))
    print('  ===================================')
    
    print('\033[1m' + '\n Predicted value (skflow.DNNClassifier) = ' + predValue +
          '  (probability = ' + str(predProb) + '%)\033[0m\n')

    return predValue, predProb
