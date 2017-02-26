#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict
* Perform Machine Learning on Raman spectra.
* version: 20170226a
*
* Uses: SVM, Neural Networks, TensorFlow, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
print(__doc__)

import matplotlib
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Agg')

import numpy as np
import sys, os.path, getopt, glob, csv
from os.path import exists, splitext
from os import rename
from datetime import datetime, date
import random

#**********************************************
''' Calculation by limited number of points '''
#**********************************************
cherryPickEnPoint = False  # False recommended

enSel = [1050, 1150, 1220, 1270, 1330, 1410, 1480, 1590, 1620, 1650]
enSelDelta = [2, 2, 2, 2, 10, 2, 2, 15, 5, 2]

if(cherryPickEnPoint == True):
    enRestrictRegion = False
    print(' Calculation by limited number of points: ENABLED ')
    print(' THIS IS AN EXPERIMENTAL FEATURE \n')
    print(' Restricted range: DISABLED')

#**********************************************
''' Spectra normalization, preprocessing '''
#**********************************************
Ynorm = True   # True recommended
YnormTo = 1
YnormX = 1600
YnormXdelta = 30

fullYnorm = False  # Normalize full spectra (False: recommended)

preProcess = True  # True recommended

enRestrictRegion = False
enLim1 = 450    # for now use indexes rather than actual Energy
enLim2 = 550    # for now use indexes rather than actual Energy

scrambleNoise = False # Adds random noise to spectra (False: recommended)
scrambleNoiseOffset = 0.1

#**********************************************
''' Model selection for training '''
#**********************************************
modelSelection = False
percentCrossValid = 0.05

#**********************************************
''' Support Vector Classification'''
#**********************************************
runSVM = True
svmClassReport = False

class svmDef:
    svmAlwaysRetrain = False
    plotSVM = True

''' Training algorithm for SVM
Use either 'linear' or 'rbf'
('rbf' for large number of features) '''

Cfactor = 20
kernel = 'rbf'
showClasses = False

#**********************************************
''' Neural Networks'''
#**********************************************
runNN = True
nnClassReport = False

#nnTrainedData = "nnModel.pkl"
class nnDef:
    nnAlwaysRetrain = False
    plotNN = True

''' Solver for NN
    lbfgs preferred for small datasets
    (alternatives: 'adam' or 'sgd') '''
nnSolver = 'lbfgs'
nnNeurons = 100  #default = 100

#**********************************************
''' TensorFlow '''
#**********************************************
runTF = True

class tfDef:
    tfAlwaysRetrain = False
    tfAlwaysImprove = False # tfAlwaysRetrain must be True for this to work
    plotTF = True
    
    singleIter = True
    percentTFCrossValid = 0.3
    trainingIter = 1000

#**********************************************
''' Principal component analysis (PCA) '''
#**********************************************
runPCA = False
customNumPCAComp = True
numPCAcomponents = 2

#**********************************************
''' K-means '''
#**********************************************
runKM = False
customNumKMComp = False
numKMcomponents = 20

class kmDef:
    plotKM = True
    plotKMmaps = True

#**********************************************
''' Plotting '''
#**********************************************
showProbPlot = False
showTrainingDataPlot = False
showPCAPlots = True

#**********************************************
''' Multiprocessing '''
#**********************************************
multiproc = False

#**********************************************
''' Main '''
#**********************************************
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ftmbkph:", ["file", "traintf", "map", "batch", "kmaps", "pca", "help"])
    except:
        usage()
        sys.exit(2)

    if opts == []:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-f" , "--file"):
            try:
                LearnPredictFile(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)
                
        if o in ("-t" , "--traintf"):
            try:
                TrainTF(sys.argv[2])
            except:
                usage()
                sys.exit(2)
        
        if o in ("-m" , "--map"):
            try:
                LearnPredictMap(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)

        if o in ("-b" , "--batch"):
            #try:
            LearnPredictBatch(sys.argv[2])
                #except:
                #usage()
                #sys.exit(2)

        if o in ("-p" , "--pca"):
            try:
                PCA(sys.argv[2])
            except:
                usage()
                sys.exit(2)

        if o in ("-k" , "--kmaps"):
            try:
                if len(sys.argv) > 3:
                    numKMcomp = sys.argv[3]
                else:
                    numKMcomp = numKMcomponents
                KmMap(sys.argv[2], numKMcomp)
            except:
                usage()
                sys.exit(2)

#**********************************************
''' Learn and Predict - File'''
#**********************************************
def LearnPredictFile(learnFile, sampleFile):
    
    ''' Open and process training data '''
    En, Cl, A, Amax, YnormXind = readLearnFile(learnFile)
    
    learnFileRoot = os.path.splitext(learnFile)[0]

    ''' Run PCA '''
    if runPCA == True:
        runPCAmain(A, Cl, En)

    ''' Open prediction file '''
    R, Rx = readPredFile(sampleFile)
    
    ''' Preprocess prediction data '''
    A, Cl, En, R, Aorig, Rorig = preProcessNormData(R, Rx, A, En, Cl, Amax, YnormXind, 0)
    
    ''' Run Support Vector Machines '''
    if runSVM == True:
        runSVMmain(A, Cl, En, R, learnFileRoot)

    ''' Run Neural Network '''
    if runNN == True:
        runNNmain(A, Cl, R, learnFileRoot)

    ''' Tensorflow '''
    if runTF == True:
        runTensorFlow(A,Cl,R, learnFileRoot)

    ''' Plot Training Data '''
    if showTrainingDataPlot == True:
        plotTrainData(A, En, R)

    ''' Run K-Means '''
    if runKM == True:
        runKMmain(A, Cl, En, R, Aorig, Rorig)


#**********************************************
''' Process - Batch'''
#**********************************************

def processSingleBatch(f, En, Cl, A, Amax, YnormXind, summary_filename, learnFile):
    print(' Processing file: \033[1m' + f + '\033[0m\n')
    R, Rx = readPredFile(f)
    summaryFile = [f]
    ''' Preprocess prediction data '''
    A, Cl, En, R, Aorig, Rorig = preProcessNormData(R, Rx, A, En, Cl, Amax, YnormXind, 0)
    
    learnFileRoot = os.path.splitext(learnFile)[0]
            
    ''' Run Support Vector Machines '''
    if runSVM == True:
        svmPred, svmProb = runSVMmain(A, Cl, En, R, learnFileRoot)
        summaryFile.extend([svmPred, svmProb])
        svmDef.svmAlwaysRetrain = False
            
    ''' Run Neural Network '''
    if runNN == True:
        nnPred, nnProb = runNNmain(A, Cl, R, learnFileRoot)
        summaryFile.extend([nnPred, nnProb])
        nnDef.nnAlwaysRetrain = False
            
    ''' Tensorflow '''
    if runTF == True:
        tfPred, tfProb = runTensorFlow(A,Cl,R, learnFileRoot)
        summaryFile.extend([tfPred, tfProb])
        tfDef.tfAlwaysRetrain = False
    
    ''' Run K-Means '''
    if runKM == True:
        kmDef.plotKM = False
        kmPred = runKMmain(A, Cl, En, R, Aorig, Rorig)
        summaryFile.extend([kmPred])
            
    with open(summary_filename, "a") as sum_file:
        csv_out=csv.writer(sum_file)
        csv_out.writerow(summaryFile)
        sum_file.close()


#**********************************************
''' Learn and Predict - Maps'''
#**********************************************
def LearnPredictMap(learnFile, mapFile):
    
    ''' Open and process training data '''
    En, Cl, A, Amax, YnormXind = readLearnFile(learnFile)
    
    learnFileRoot = os.path.splitext(learnFile)[0]

    ''' Open prediction map '''
    X, Y, R, Rx = readPredMap(mapFile)
    type = 0
    i = 0;
    svmPred = nnPred = tfPred = kmPred = np.empty([X.shape[0]])
    print(' Processing map...' )
    for r in R[:]:
        A, Cl, En, r, Aorig, rorig = preProcessNormData(r, Rx, A, En, Cl, Amax, YnormXind, type)
        type = 1

        ''' Run Support Vector Machines '''
        if runSVM == True:
            svmPred[i], temp = runSVMmain(A, Cl, En, r, learnFileRoot)
            saveMap(mapFile, 'svm', 'HC', svmPred[i], X[i], Y[i], True)
            svmDef.svmAlwaysRetrain = False
    
        ''' Run Neural Network '''
        if runNN == True:
            nnPred[i], temp = runNNmain(A, Cl, r, learnFileRoot)
            saveMap(mapFile, 'NN', 'HC', nnPred[i], X[i], Y[i], True)
            nnDef.nnAlwaysRetrain = False
    
        ''' Tensorflow '''
        if runTF == True:
            tfPred[i], temp = runTensorFlow(A,Cl,r, learnFileRoot)
            saveMap(mapFile, 'TF', 'HC', tfPred[i], X[i], Y[i], True)
            tfDef.tfAlwaysRetrain = False
        
        ''' Run K-Means '''
        if runKM == True:
            kmDef.plotKM = False
            kmPred[i] = runKMmain(A, Cl, En, r, Aorig, rorig)
            saveMap(mapFile, 'KM', 'HC', kmPred[i], X[i], Y[i], True)
        i+=1

    if svmDef.plotSVM == True and runSVM == True:
        plotMaps(X, Y, svmPred, 'SVM')
    if nnDef.plotNN == True and runNN == True:
        plotMaps(X, Y, nnPred, 'Neural netowrks')
    if tfDef.plotTF == True and runTF == True:
        plotMaps(X, Y, tfPred, 'TensorFlow')
    if kmDef.plotKMmaps == True and runKM == True:
        plotMaps(X, Y, kmPred, 'K-Means Prediction')

#********************************************************************************
''' Setup training-only via TensorFlow '''
#********************************************************************************
def TrainTF(learnFile):
    ''' Open and process training data '''
    En, Cl, A, Amax, YnormXind = readLearnFile(learnFile)
    ''' Preprocess prediction data '''
    A, Cl, En, R, Aorig, Rorig = preProcessNormData(A[random.randint(1,A.shape[0]),:], En, A, En, Cl, Amax, YnormXind, 0)
    print(" Using random spectra from training dataset as evaluation file ")
    ''' Tensorflow '''
    learnFileRoot = os.path.splitext(learnFile)[0]
    runTensorFlow(A,Cl,R,learnFileRoot)


#**********************************************
''' Setup and Run PCA'''
#**********************************************
def PCA(learnFile):
    ''' Open and process training data '''
    En, Cl, A, Amax, YnormXind = readLearnFile(learnFile)
    ''' Run PCA '''
    runPCAmain(A, Cl, En)


#********************
''' Run SVM '''
#********************
def runSVMmain(A, Cl, En, R, Root):
    from sklearn import svm
    from sklearn.externals import joblib
    svmTrainedData = Root + '.svmModel.pkl'
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
def runNNmain(A, Cl, R, Root):
    from sklearn.neural_network import MLPClassifier
    from sklearn.externals import joblib
    nnTrainedData = Root + '.nnModel.pkl'
    
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
''' https://www.tensorflow.org/get_started/mnist/beginners'''
#********************************************************************************

#********************************************************************************
''' Format vectors of unique labels '''
#********************************************************************************
def formatClass(formatClassfile, Cl):
    try:
        with open(formatClassfile) as f:
            print('\n Opening TensorFlow format class data...')
            Cl2 = np.loadtxt(f, unpack =False)
            if len(Cl2) != len(Cl):
                raise ValueError(' WARNING: format class has different length')
    except:
        print( '\n Formatting training cluster data...')
        Cl2 = np.zeros((np.array(Cl).shape[0], np.unique(Cl).shape[0]))
        for i in range(np.array(Cl).shape[0]):
            for j in range(np.unique(Cl).shape[0]):
                if np.array(Cl)[i] == np.unique(Cl)[j]:
                    Cl2[i,j] = 1
                else:
                    Cl2[i,j] = 0
        np.savetxt(formatClassfile, Cl2, delimiter=' ', fmt='%d')
    return Cl2


#********************************************************************************
''' Run model training and evaluation via TensorFlow '''
#********************************************************************************
def runTensorFlow(A, Cl, R, Root):
    import tensorflow as tf
    formatClassfile = Root + '.tfclass'
    tfTrainedData = Root + '.tfmodel'
    Cl2 = formatClass(formatClassfile, Cl)


    print(' Initializing TensorFlow...\n')
    x = tf.placeholder(tf.float32, [None, A.shape[1]])
    W = tf.Variable(tf.zeros([A.shape[1], np.unique(Cl).shape[0]]))
    b = tf.Variable(tf.zeros(np.unique(Cl).shape[0]))
    y_ = tf.placeholder(tf.float32, [None, np.unique(Cl).shape[0]])

    # The raw formulation of cross-entropy can be numerically unstable
    #y = tf.nn.softmax(tf.matmul(x, W) + b)
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))

    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    y = tf.matmul(x,W) + b
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    try:
        if tfDef.tfAlwaysRetrain == False:
            print('\n Opening TF training model from:', tfTrainedData)
            saver.restore(sess, './' + tfTrainedData)
            print(' Model restored.')
        else:
            raise ValueError(' Force TF model retraining.')
    except:
        init = tf.global_variables_initializer()
        sess.run(init)

        if os.path.isfile(tfTrainedData + '.meta') & tfDef.tfAlwaysImprove == True:
            print('\n Improving TF model...')
            saver.restore(sess, './' + tfTrainedData)
        else:
            print('\n Rebuildind TF model...')

        if tfDef.singleIter == True:
            print(' Iterating training using subset (' +  str(tfDef.percentTFCrossValid*100) + '%), ' + str(tfDef.trainingIter) + ' times ...')
            for i in range(tfDef.trainingIter):
                As, Cl2s = formatSubset(A, Cl2, tfDef.percentTFCrossValid)
                sess.run(train_step, feed_dict={x: As, y_: Cl2s})
        else:
                sess.run(train_step, feed_dict={x: A, y_: Cl2})

        save_path = saver.save(sess, tfTrainedData)
        print(' Model saved in file: %s' % save_path)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    res1 = sess.run(y, feed_dict={x: R})
    res2 = sess.run(tf.argmax(y, 1), feed_dict={x: R})
    
    print(' Accuracy: ' + str(100*sess.run(accuracy, feed_dict={x: R, y_: Cl2})) + '%\n')
    print('\033[1m' + ' Predicted value (TF): ' + str(np.unique(Cl)[res2][0]) + ' (Probability: ' + str('{:.1f}'.format(res1[0][res2][0])) + '%)\n' + '\033[0m' )
    return np.unique(Cl)[res2][0], res1[0][res2][0]


#********************************************************************************
''' Run PCA '''
''' Transform data:
    pca.fit(data).transform(data)
    Loading Vectors (eigenvectors):
    pca.components_
    Eigenvalues:
    pca.explained_variance_ratio
    '''
#********************************************************************************
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


#************************************
''' Read Learning file '''
#************************************
def readLearnFile(learnFile):
    try:
        with open(learnFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Map data file not found \n' + '\033[0m')
        return
    
    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(M,np.s_[0:1],0)
    Cl = ['{:.2f}'.format(x) for x in M[:,0]]
    A = np.delete(M,np.s_[0:1],1)
    Atemp = A[:,range(len(enSel))]

    if cherryPickEnPoint == True and enRestrictRegion == False:
        enPoints = list(enSel)
        enRange = list(enSel)

        for i in range(0, len(enSel)):
            enRange[i] = np.where((En<float(enSel[i]+enSelDelta[i])) & (En>float(enSel[i]-enSelDelta[i])))[0].tolist()

            for j in range(0, A.shape[0]):
                Atemp[j,i] = A[j,A[j,enRange[i]].tolist().index(max(A[j, enRange[i]].tolist()))+enRange[i][0]]
            
            enPoints[i] = int(np.average(enRange[i]))
        A = Atemp
        En = En[enPoints]

        if type == 0:
            print( ' Cheery picking points in the spectra\n')

    # Find index corresponding to energy value to be used for Y normalization
    if fullYnorm == False:
        YnormXind = np.where((En<float(YnormX+YnormXdelta)) & (En>float(YnormX-YnormXdelta)))[0].tolist()
    else:
        YnormXind = np.where(En>0)[0].tolist()
    Amax = np.empty([A.shape[0],1])

    print(' Number of datapoints = ' + str(A.shape[0]))
    print(' Size of each datapoint = ' + str(A.shape[1]) + '\n')
    return En, Cl, A, Amax, YnormXind

#**********************************************
''' Open prediction file '''
#**********************************************
def readPredFile(sampleFile):
    try:
        with open(sampleFile, 'r') as f:
            print(' Opening sample data for prediction...')
            Rtot = np.loadtxt(f, unpack =True)
    except:
        print('\033[1m' + '\n Sample data file not found \n ' + '\033[0m')
        return

    R=Rtot[1,:]
    Rx=Rtot[0,:]

    if cherryPickEnPoint == True and enRestrictRegion == False:
        Rtemp = R[range(len(enSel))]
        enPoints = list(enSel)
        enRange = list(enSel)
        for i in range(0, len(enSel)):
            enRange[i] = np.where((Rx<float(enSel[i]+enSelDelta[i])) & (Rx>float(enSel[i]-enSelDelta[i])))[0].tolist()
            Rtemp[i] = R[R[enRange[i]].tolist().index(max(R[enRange[i]].tolist()))+enRange[i][0]]

            enPoints[i] = int(np.average(enRange[i]))
        R = Rtemp
        Rx = Rx[enPoints]

    return R, Rx


#**********************************************
''' Learn and Predict - Batch'''
#**********************************************
def LearnPredictBatch(learnFile):
    summary_filename = 'summary' + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
    makeHeaderSummary(summary_filename, learnFile)
    ''' Open and process training data '''
    En, Cl, A, Amax, YnormXind = readLearnFile(learnFile)
    if multiproc == True:
        from multiprocessing import Pool
        import multiprocessing as mp
        p = mp.Pool()
        for f in glob.glob('*.txt'):
            if (f != learnFile):
                p.apply_async(processSingleBatch, args=(f, En, Cl, A, Amax, YnormXind, summary_filename, learnFile))
        p.close()
        p.join()
    else:
        for f in glob.glob('*.txt'):
            if (f != learnFile):
                processSingleBatch(f, En, Cl, A, Amax, YnormXind, summary_filename, learnFile)


#**********************************************************************************
''' Preprocess prediction data '''
#**********************************************************************************
def preProcessNormData(R, Rx, A, En, Cl, Amax, YnormXind, type):
    #**********************************************************************************
    ''' Reformat x-axis in case it does not match that of the training data '''
    #**********************************************************************************
    if(R.shape[0] != A.shape[1]):
        if type == 0:
            print('\033[1m' + '\n WARNING: Different number of datapoints for the x-axis\n for training (' + str(A.shape[1]) + ') and sample (' + str(R.shape[0]) + ') data.\n Reformatting x-axis of sample data...\n' + '\033[0m')
        R = np.interp(En, Rx, R)
    R = R.reshape(1,-1)

    if scrambleNoise == True:
        scrambleNoise(A, scambleNoiseOffset)
    Aorig = np.copy(A)
    Rorig = np.copy(R)

    #**********************************************
    ''' Normalize/preprocess if flags are set '''
    #**********************************************
    if Ynorm == True:
        if type == 0:
            print(' Normalizing spectral intensity to: ' + str(YnormTo) + '; En = [' + str(YnormX-YnormXdelta) + ', ' + str(YnormX+YnormXdelta) + ']')
        for i in range(0,A.shape[0]):
            Amax[i] = A[i,A[i][YnormXind].tolist().index(max(A[i][YnormXind].tolist()))+YnormXind[0]]
            A[i,:] = np.multiply(A[i,:], YnormTo/Amax[i])
        Rmax = R[0,R[0][YnormXind].tolist().index(max(R[0][YnormXind].tolist()))+YnormXind[0]]
        R[0,:] = np.multiply(R[0,:], YnormTo/Rmax)

    if preProcess == True:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(A)
        A = scaler.transform(A)
        R = scaler.transform(R)
    
    #**********************************************
    ''' Select subset of training data for cross validation '''
    #**********************************************
    if modelSelection == True:
        A, Cl = formatSubset(A, Cl, percentCrossValid)

    #**********************************************
    ''' Energy normalization range '''
    #**********************************************
    if enRestrictRegion == True:
        A = A[:,range(enLim1, enLim2)]
        En = En[range(enLim1, enLim2)]
        R = R[:,range(enLim1, enLim2)]
        Aorig = Aorig[:,range(enLim1, enLim2)]
        Rorig = Rorig[:,range(enLim1, enLim2)]

        if type == 0:
            print( ' Restricting energy range between: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    else:
        if type == 0:
            if(cherryPickEnPoint == True):
                print( ' Using selected spectral points:')
                print(En)
            else:
                print( ' Using full energy range: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')

    return A, Cl, En, R, Aorig, Rorig


#**********************************************************************************
''' Preprocess prediction data '''
#**********************************************************************************
def preProcessNormMap(A, En, type):
    #**********************************************************************************
    ''' Reformat x-axis in case it does not match that of the training data '''
    #**********************************************************************************
    
    # Find index corresponding to energy value to be used for Y normalization
    if fullYnorm == False:
        YnormXind = np.where((En<float(YnormX+YnormXdelta)) & (En>float(YnormX-YnormXdelta)))[0].tolist()
    else:
        YnormXind = np.where(En>0)[0].tolist()
    
    Amax = np.empty([A.shape[0],1])
    Aorig = np.copy(A)
    
    #**********************************************
    ''' Normalize/preprocess if flags are set '''
    #**********************************************
    if Ynorm == True:
        if type == 0:
            print(' Normalizing spectral intensity to: ' + str(YnormTo) + '; En = [' + str(YnormX-YnormXdelta) + ', ' + str(YnormX+YnormXdelta) + ']')
        for i in range(0,A.shape[0]):
            Amax[i] = A[i,A[i][YnormXind].tolist().index(max(A[i][YnormXind].tolist()))+YnormXind[0]]
            A[i,:] = np.multiply(A[i,:], YnormTo/Amax[i])


    if preProcess == True:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(A)
        A = scaler.transform(A)
    
    #**********************************************
    ''' Energy normalization range '''
    #**********************************************
    if enRestrictRegion == True:
        A = A[:,range(enLim1, enLim2)]
        En = En[range(enLim1, enLim2)]
        Aorig = Aorig[:,range(enLim1, enLim2)]
        if type == 0:
            print( ' Restricting energy range between: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    else:
        if type == 0:
            print( ' Using full energy range: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    
    return A, En, Aorig


####################################################################
''' Format subset of training data '''
####################################################################
def formatSubset(A, Cl, percent):
    from sklearn.model_selection import train_test_split
    A_train, A_cv, Cl_train, Cl_cv = \
    train_test_split(A, Cl, test_size=percent, random_state=42)
    return A_train, Cl_train


####################################################################
''' Open map files '''
####################################################################
def readPredMap(mapFile):
    try:
        with open(mapFile, 'r') as f:
            En = np.array(f.readline().split(), dtype=np.dtype(float))
            A = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Map data file not found \n' + '\033[0m')
        return

    X = A[:,0]
    Y = A[:,1]
    A = np.delete(A, np.s_[0:2], 1)
    print(' Shape map: ' + str(A.shape))
    return X, Y, A, En


####################################################################
''' Save map files '''
####################################################################
def saveMap(file, type, extension, s, x1, y1, comma):
    
    inputFile = saveMapName(file, type, extension, comma)
    
    with open(inputFile, "a") as coord_file:
        if comma==True:
            coord_file.write('{:},'.format(x1))
            coord_file.write('{:},'.format(y1))
        else:
            coord_file.write('{:}\t'.format(x1))
            coord_file.write('{:}\t'.format(y1))
        coord_file.write('{:}\n'.format(s))
        coord_file.close()

def saveMapName(file, type, extension, comma):
    if comma==True:
        extension2 = '_map.csv'
    else:
        extension2 = '_map.txt'
    return os.path.splitext(file)[0] + '_' + type + '-' + extension + extension2


#************************************
''' Plot Probabilities'''
#************************************
def plotProb(clf, R):
    prob = clf.predict_proba(R)[0].tolist()
    print(' Probabilities of this sample within each class: \n')
    for i in range(0,clf.classes_.shape[0]):
        print(' ' + str(clf.classes_[i]) + ': ' + str(round(100*prob[i],2)) + '%')
    import matplotlib.pyplot as plt
    print('\n Stand by: Plotting probabilities for each class... \n')
    plt.title('Probability density per class')
    for i in range(0, clf.classes_.shape[0]):
        plt.scatter(clf.classes_[i], round(100*prob[i],2), label='probability', c = 'red')
    plt.grid(True)
    plt.xlabel('Class')
    plt.ylabel('Probability [%]')
    plt.show()


#************************************
''' Plot Training data'''
#************************************
def plotTrainData(A, En, R):
    import matplotlib.pyplot as plt
    print(' Stand by: Plotting each datapoint from the map...\n')
    if Ynorm ==True:
        plt.title("Normalized Training Data")
    else:
        plt.title("Training Data")
    for i in range(0,A.shape[0]):
        plt.plot(En, A[i,:], label='Training data')
    plt.plot(En, R[0,:], linewidth = 4, label='Sample data')
    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')
    plt.show()


#************************************
''' Plot Processed Maps'''
#************************************
def plotMaps(X, Y, A, label):
    print(' Plotting ' + label + ' Map...\n')
    import scipy.interpolate
    xi = np.linspace(min(X), max(X))
    yi = np.linspace(min(Y), max(Y))
    xi, yi = np.meshgrid(xi, yi)
    
    rbf = scipy.interpolate.Rbf(Y, -X, A, function='linear')
    zi = rbf(xi, yi)
    import matplotlib.pyplot as plt
    plt.imshow(zi, vmin=A.min(), vmax=A.max(), origin='lower',label='data',
               extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.title(label)
    plt.xlabel('X [um]')
    plt.ylabel('Y [um]')
    plt.show()


####################################################################
''' Make header, if absent, for the summary file '''
####################################################################
def makeHeaderSummary(file, learnFile):
    if os.path.isfile(file) == False:
        summaryHeader1 = ['Training File:', learnFile]
        summaryHeader2 = ['File','SVM-HC','SVM-Prob%', 'NN-HC', 'NN-Prob%', 'TF-HC', 'TF-Prob%']
        with open(file, "a") as sum_file:
            csv_out=csv.writer(sum_file)
            csv_out.writerow(summaryHeader1)
            csv_out.writerow(summaryHeader2)
            sum_file.close()


#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:')
    print(' single files: ')
    print('  python SpectraLearnPredictSVM.py -f <learningfile> <spectrafile> \n')
    print(' maps (formatted for Horiba LabSpec): ')
    print('  python SpectraLearnPredictSVM.py -m <learningfile> <spectramap> \n')
    print(' batch txt files: ')
    print('  python SpectraLearnPredictSVM.py -b <learningfile> \n')
    print(' k-means on maps: ')
    print('  python SpectraLearnPredictSVM.py -k <spectramap> <number_of_classes>\n')
    print(' Principal component analysis only: ')
    print('  python SpectraLearnPredictSVM.py -p <spectramap>\n')


#************************************
''' Info on Classification Report '''
#************************************
def runClassReport(clf, A, Cl):
    from sklearn.metrics import classification_report
    y_pred = clf.predict(A)
    print(classification_report(Cl, y_pred, target_names=clf.classes_))
    print(' Precision is the probability that, given a classification result for a sample,\n' +
          ' the sample actually belongs to that class. Recall (Accuracy) is the probability that a \n' +
          ' sample will be correctly classified for a given class. F1 score combines both \n' +
          ' accuracy and precision to give a single measure of relevancy of the classifier results.\n')


#************************************
''' Introduce Noise in Data '''
#************************************
def scrambleNoise(A, offset):
    from random import uniform
    for i in range(A.shape[1]):
        A[:,i] += offset*uniform(-1,1)


#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
