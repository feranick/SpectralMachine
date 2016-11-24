#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* ML - SpectraLearnPredict
* Perform Machine Learning on Raman data/maps.
* version: 20161116c
*
* Uses: SVM, Neural Networks, TensorFlow, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''

import matplotlib
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Agg')

import numpy as np
import sys, os.path, getopt, glob, csv
from os.path import exists
from os import rename
from datetime import datetime, date

from config_SLP import *

#********************
''' Run SVM '''
#********************
def runSVMmain(A, Cl, En, R):
    from sklearn import svm
    from sklearn.externals import joblib
    print('\n Running Support Vector Machine (kernel: ' + kernel + ')...')
    try:
        if svmDef.svmAlwaysRetrain == False:
            with open(svmTrainedData):
                print(' Opening SVM training model...')
                clf = joblib.load(svmTrainedData)
        else:
            raise ValueError('Force retraining SVM model')
    except:
        #**********************************************
        ''' Retrain NN data if not available'''
        #**********************************************
        print(' Retraining SVM data...')
        clf = svm.SVC(C = Cfactor, decision_function_shape = 'ovr', probability=True)
        clf.fit(A,Cl)
        Z= clf.decision_function(A)
        print(' Number of classes = ' + str(Z.shape[1]))
        joblib.dump(clf, svmTrainedData)
        if showClasses == True:
            print(' List of classes: ' + str(clf.classes_))

    R_pred = clf.predict(R)
    prob = clf.predict_proba(R)[0].tolist()
    print('\033[1m' + '\n Predicted value (SVM) = ' + str(R_pred[0]) + '\033[0m' + ' (probability = ' +
          str(round(100*max(prob),1)) + '%)\n')
          
    #**************************************
    ''' SVM Classification Report '''
    #**************************************
    if svmClassReport == True:
        print(' SVM Classification Report \n')
        runClassReport(clf, A, Cl)

    #*************************
    ''' Plot probabilities '''
    #*************************
    if showProbPlot == True:
        plotProb(clf, R)

    return R_pred[0], round(100*max(prob),1)


#*************************
''' Run Neural Network '''
#*************************
def runNNmain(A, Cl, R):
    from sklearn.neural_network import MLPClassifier
    from sklearn.externals import joblib
    print('\n Running Neural Network: multi-layer perceptron (MLP) - (solver: ' + nnSolver + ')...')
    try:
        if nnDef.nnAlwaysRetrain == False:
            with open(nnTrainedData):
                print(' Opening NN training model...')
                clf = joblib.load(nnTrainedData)
        else:
            raise ValueError('Force NN retraining.')
    except:
        #**********************************************
        ''' Retrain data if not available'''
        #**********************************************
        print(' Retraining NN model...')
        clf = MLPClassifier(solver=nnSolver, alpha=1e-5, hidden_layer_sizes=(nnNeurons,), random_state=1)
        clf.fit(A, Cl)
        joblib.dump(clf, nnTrainedData)

    prob = clf.predict_proba(R)[0].tolist()
    print('\033[1m' + '\n Predicted value (Neural Networks) = ' + str(clf.predict(R)[0]) + '\033[0m' +
          ' (probability = ' + str(round(100*max(prob),1)) + '%)\n')

    #**************************************
    ''' Neural Networks Classification Report '''
    #**************************************
    if nnClassReport == True:
        print(' Neural Networks Classification Report\n')
        runClassReport(clf, A, Cl)

    #*************************
    ''' Plot probabilities '''
    #*************************
    if showProbPlot == True:
        plotProb(clf, R)

    return clf.predict(R)[0], round(100*max(prob),1)


#********************************************************************************
''' Tensorflow '''
''' https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html'''
#********************************************************************************
def runTensorFlow(A, Cl, R):
    import tensorflow as tf
    formatClassfile = "tfFormatClass.txt"
    
    try:
        with open(formatClassfile) as f:
            print('\n Opening TensorFlow format class data...')
            Cl2 = np.loadtxt(f, unpack =False)
    except:
        print( '\n Formatting training cluster data...')
        Cl2 = np.zeros((np.array(Cl).shape[0], np.unique(Cl).shape[0]))
        
        for i in range(np.array(Cl).shape[0]):
            for j in range(np.array(np.unique(Cl).shape[0])):
                if np.array(Cl)[i] == np.unique(Cl)[j]:
                    np.put(Cl2, [i, j], 1)
                else:
                    np.put(Cl2, [i, j], 0)
                
                np.savetxt(formatClassfile, Cl2, delimiter='\t', fmt='%10.6f')

    print(' Initializing TensorFlow...')
    x = tf.placeholder(tf.float32, [None, A.shape[1]])
    W = tf.Variable(tf.zeros([A.shape[1], np.unique(Cl).shape[0]]))
    b = tf.Variable(tf.zeros(np.unique(Cl).shape[0]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    y_ = tf.placeholder(tf.float32, [None, np.unique(Cl).shape[0]])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    try:
        if tfDef.tfAlwaysRetrain == False:
            with open(tfTrainedData):
                print(' Opening TF training model from:', tfTrainedData)
                saver = tf.train.Saver()
                sess = tf.Session()
                saver.restore(sess, tfTrainedData)
                print(' Model restored.')
        else:
            raise ValueError(' Force TF model retraining.')
    except:
        print(' Improving TF model...')
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init)
        sess.run(train_step, feed_dict={x: A, y_: Cl2})

        save_path = saver.save(sess, tfTrainedData)
        print(' Model saved in file: %s' % save_path)

    res1 = sess.run(y, feed_dict={x: R})
    res2 = sess.run(tf.argmax(y, 1), feed_dict={x: R})
    
    print(' Accuracy: ' + str(sess.run(accuracy, feed_dict={x: R, y_: Cl2})) + '%\n')
    print('\033[1m' + ' Predicted value (TF): ' + str(np.unique(Cl)[res2][0]) + ' (' + str('{:.1f}'.format(res1[0][res2][0]*100)) + '%)\n' + '\033[0m' )
    return np.unique(Cl)[res2][0], res1[0][res2][0]*100


#********************
''' Run PCA '''
''' Transform data:
    pca.fit(data).transform(data)
    Loading Vectors (eigenvectors):
    pca.components_
    Eigenvalues:
    pca.explained_variance_ratio
    '''
#********************
def runPCAmain(A, Cl, En):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    print(' Running PCA...\n')
    print(' Number of unique identifiers in training data: ' + str(np.unique(Cl).shape[0]))
    if customNumPCAComp == False:
        numPCAcomp = np.unique(Cl).shape[0]
    else:
        numPCAcomp = numPCAcomponents
    print(' Number of Principal components: ' + str(numPCAcomp) + '\n')
    pca = PCA(n_components=numPCAcomp)
    A_r = pca.fit(A).transform(A)


    for i in range(0,pca.components_.shape[0]):
        print(' Score PCA ' + str(i) + ': ' + '{0:.0f}%'.format(pca.explained_variance_ratio_[i] * 100))
    print('')

    if showPCAPlots == True:
        print(' Plotting Loadings and score plots... \n')
        
        #***************************
        ''' Plotting Loadings '''
        #***************************
        for i in range(0,pca.components_.shape[0]):
            plt.plot(En, pca.components_[i,:], label='PC' + str(i) + ' ({0:.0f}%)'.format(pca.explained_variance_ratio_[i] * 100))
        plt.plot((En[0], En[En.shape[0]-1]), (0.0, 0.0), 'k--')
        plt.title('Loadings plot')
        plt.xlabel('Raman shift [1/cm]')
        plt.ylabel('Principal component')
        plt.legend()
        plt.figure()

        #***************************
        ''' Plotting Scores '''
        #***************************
        Cl_ind = np.zeros(len(Cl))
        Cl_labels = np.zeros(0)
        ind = np.zeros(np.unique(Cl).shape[0])

        for i in range(len(Cl)):
            if (np.in1d(Cl[i], Cl_labels, invert=True)):
                Cl_labels = np.append(Cl_labels, Cl[i])

        for i in range(len(Cl)):
            Cl_ind[i] = np.where(Cl_labels == Cl[i])[0][0]

            colors = [ cm.jet(x) for x in np.linspace(0, 1, ind.shape[0]) ]

        for color, i, target_name in zip(colors, range(ind.shape[0]), Cl_labels):
            plt.scatter(A_r[Cl_ind==i,0], A_r[Cl_ind==i,1], color=color, alpha=.8, lw=2, label=target_name)

        plt.title('Score plot')
        plt.xlabel('PC 0 ({0:.0f}%)'.format(pca.explained_variance_ratio_[0] * 100))
        plt.ylabel('PC 1 ({0:.0f}%)'.format(pca.explained_variance_ratio_[1] * 100))
        plt.figure()

        #******************************
        ''' Plotting Scores vs H:C '''
        #******************************
        for j in range(2):
            for color, i, target_name in zip(colors, range(ind.shape[0]), Cl_labels):
                plt.scatter(np.asarray(Cl)[Cl_ind==i], A_r[Cl_ind==i,j], color=color, alpha=.8, lw=2, label=target_name)
            plt.xlabel('H:C elemental ratio')
            plt.ylabel('PC ' + str(j) + ' ({0:.0f}%)'.format(pca.explained_variance_ratio_[j] * 100))
            plt.figure()
        plt.show()

#********************
''' Run K-Means '''
#********************
def runKMmain(A, Cl, En, R, Aorig, Rorig):
    from sklearn.cluster import KMeans
    print('\n Running K-Means...')
    print(' Number of unique identifiers in training data: ' + str(np.unique(Cl).shape[0]))
    if customNumKMComp == False:
        numKMcomp = np.unique(Cl).shape[0]
    else:
        numKMcomp = numKMcomponents
    kmeans = KMeans(n_clusters=numKMcomp, random_state=0).fit(A)
    for i in range(0, numKMcomp):
        print('\n Class: ' + str(i) + '\n  ',end="")
        for j in range(0,kmeans.labels_.shape[0]):
            if kmeans.labels_[j] == i:
                print(' ' + str(Cl[j]), end="")
    print('\033[1m' + '\n\n Predicted class (K-Means) = ' + str(kmeans.predict(R)[0]) + '\033[0m \n')
    if kmDef.plotKM == True:
        import matplotlib.pyplot as plt
        for j in range(0,kmeans.labels_.shape[0]):
            if kmeans.labels_[j] == kmeans.predict(R)[0]:
                plt.plot(En, Aorig[j,:])
        plt.plot(En, Rorig[0,:], linewidth = 2, label='Predict')
        plt.title('K-Means')
        plt.xlabel('Raman shift [1/cm]')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()
    return kmeans.predict(R)[0]

#**********************************************
''' K-Means - Maps'''
#**********************************************
def KmMap(mapFile, numKMcomp):
    
    ''' Open prediction map '''
    X, Y, R, Rx = readPredMap(mapFile)
    type = 0
    i = 0;
    
    R, Rx, Rorig = preProcessNormMap(R, Rx, type)
    
    from sklearn.cluster import KMeans
    print(' Running K-Means...')
    print(' Number of classes: ' + str(numKMcomp))
    kmeans = KMeans(n_clusters=numKMcomponents, random_state=0).fit(R)
    
    kmPred = np.empty([R.shape[0]])
    
    for i in range(0, R.shape[0]):
        kmPred[i] = kmeans.predict(R[i,:].reshape(1,-1))[0]
        saveMap(mapFile, 'KM', 'Class', int(kmPred[i]), X[i], Y[i], True)
        if kmPred[i] in kmeans.labels_:
            if os.path.isfile(saveMapName(mapFile, 'KM', 'Class_'+ str(int(kmPred[i]))+'-'+str(np.unique(kmeans.labels_).shape[0]), False)) == False:
                saveMap(mapFile, 'KM', 'Class_'+ str(int(kmPred[i])) + '-'+str(np.unique(kmeans.labels_).shape[0]) , '\t'.join(map(str, Rx)), ' ', ' ', False)
            saveMap(mapFile, 'KM', 'Class_'+ str(int(kmPred[i])) + '-'+str(np.unique(kmeans.labels_).shape[0]) , '\t'.join(map(str, R[1,:])), X[i], Y[i], False)

    if kmDef.plotKM == True:
        plotMaps(X, Y, kmPred, 'K-Means')

