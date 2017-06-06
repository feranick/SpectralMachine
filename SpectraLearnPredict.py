#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict
* Perform Machine Learning on Raman spectra.
* version: 20170606b
*
* Uses: Deep Neural Networks, TensorFlow, SVM, PCA, K-Means
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

#***************************************************************
''' Spectra normalization, preprocessing, model selection  '''
#***************************************************************
class preprocDef:
    Ynorm = True   # Normalize spectra (True: recommended)
    fullYnorm = False  # Normalize considering full range (False: recommended)
    StandardScalerFlag = True  # Standardize features by removing the mean and scaling to unit variance (sklearn)

    YnormTo = 1
    YnormX = 1600
    YnormXdelta = 30

    enRestrictRegion = False
    enLim1 = 450    # for now use indexes rather than actual Energy
    enLim2 = 550    # for now use indexes rather than actual Energy
    
    scrambleNoiseFlag = False # Adds random noise to spectra (False: recommended)
    scrambleNoiseOffset = 0.1

    if StandardScalerFlag == True:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

    #**********************************************
    ''' Calculation by limited number of points '''
    #**********************************************
    cherryPickEnPoint = False  # False recommended

    enSel = [1050, 1150, 1220, 1270, 1330, 1410, 1480, 1590, 1620, 1650]
    enSelDelta = [2, 2, 2, 2, 10, 2, 2, 15, 5, 2]
    
    #enSel = [1220, 1270, 1590]
    #enSelDelta = [2, 2, 30]

    if(cherryPickEnPoint == True):
        enRestrictRegion = False
        print(' Calculation by limited number of points: ENABLED ')
        print(' THIS IS AN EXPERIMENTAL FEATURE \n')
        print(' Restricted range: DISABLED')

#**********************************************
''' Deep Neural Networks - sklearn'''
#**********************************************
class nnDef:
    runNN = True
    
    alwaysRetrain = False
    subsetCrossValid = False
    percentCrossValid = 0.10    # proportion of TEST data for cross validation
    iterCrossValid = 2

    numNeurons = 200            #default = 200
    
    # Optimizers: lbfgs (default), adam, ssgd
    nnOptimizer = "lbfgs"
    
    # activation functions: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    # identity, logistic (sigmoid), tanh, relu
    
    activation_function = "relu"
    
    MLPRegressor = False
    
    # threshold in % of probabilities for listing prediction results
    thresholdProbabilityPred = 0.001

    plotNN = True
    nnClassReport = False

#***********************************************************
''' Deep Neural Networks - tensorflow via DNNClassifier'''
#***********************************************************
class dnntfDef:
    runDNNTF = True
    
    alwaysRetrain = False
    subsetCrossValid = False
    percentCrossValid = 0.10  # proportion of TEST data for cross validation
    iterCrossValid = 2
    
    numNeurons = 200        # number of neurons per layer
    numHidlayers = 1        # number of hidden layer

    # Optimizers: Adagrad (recommended), Adam, Ftrl, Momentum, RMSProp, SGD
    # https://www.tensorflow.org/api_guides/python/train
    nnOptimizer = "Adagrad"
    
    # activation functions: https://www.tensorflow.org/api_guides/python/nn
    # relu, relu6, crelu, elu, softplus, softsign, dropout, bias_add
    # sigmoid, tanh
    
    activation_function = "relu"
    
    trainingSteps = 500     #number of training steps
    
    # threshold in % of probabilities for listing prediction results
    thresholdProbabilityPred = 0.01

    #*************************************************
    # Setup variables and definitions- do not change.
    #*************************************************
    hidden_layers = [numNeurons]*numHidlayers

    if runDNNTF == True:
        import tensorflow as tf
    if activation_function == "sigmoid" or activation_function == "tanh":
        actFn = "tf."+activation_function
    else:
        actFn = "tf.nn."+activation_function
    activationFn = eval(actFn)

#**********************************************
''' Support Vector Machines'''
#**********************************************
class svmDef:
    runSVM = True
    
    alwaysRetrain = False
    subsetCrossValid = False
    percentCrossValid = 0.10  # proportion of TEST data for cross validation
    iterCrossValid = 2
    
    # threshold in % of probabilities for listing prediction results
    thresholdProbabilitySVMPred = 3
    
    ''' Training algorithm for SVM
        Use either 'linear' or 'rbf'
        ('rbf' for large number of features) '''
    
    Cfactor = 20
    kernel = 'rbf'
    showClasses = False

    plotSVM = True
    svmClassReport = False

#**********************************************
''' Principal component analysis (PCA) '''
#**********************************************
class pcaDef:
    runPCA = False
    customNumPCAComp = True
    numPCAcomponents = 2

#**********************************************
''' K-means '''
#**********************************************
class kmDef:
    runKM = False
    customNumKMComp = False
    numKMcomponents = 20
    plotKM = False
    plotKMmaps = True

#**********************************************
''' TensorFlow '''
#**********************************************
class tfDef:
    runTF = False

    alwaysRetrain = False
    alwaysImprove = False       # alwaysRetrain must be True for this to work
    subsetCrossValid = True
    percentCrossValid = 0.1     # proportion of TEST data for cross validation
    iterCrossValid = 2
    
    # threshold in % of probabilities for listing prediction results
    thresholdProbabilityTFPred = 30
    
    decayLearnRate = True
    learnRate = 0.75

    plotMapTF = True
    plotClassDistribTF = False
    enableTensorboard = False

#**********************************************
''' Plotting '''
#**********************************************
class plotDef:
    showProbPlot = False
    showPCAPlots = True
    createTrainingDataPlot = False
    showTrainingDataPlot = False
    plotAllSpectra = True  # Set to false for extremely large training sets
    
    if plotAllSpectra == False:
        stepSpectraPlot = 100  # steps in the number of spectra to be plotted

#**********************************************
''' Multiprocessing '''
#**********************************************
multiproc = False

#**********************************************
''' Main '''
#**********************************************
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "fatmbkph:", ["file", "accuracy", "traintf", "map", "batch", "kmaps", "pca", "help"])
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

        if o in ("-a" , "--accuracy"):
            print('\033[1m Running in cross validation mode for accuracy determination...\033[0m\n')
            nnDef.alwaysRetrain = True
            nnDef.subsetCrossValid = True
            dnntfDef.alwaysRetrain = True
            dnntfDef.subsetCrossValid = True
            svmDef.alwaysRetrain = True
            svmDef.subsetCrossValid = True
            tfDef.alwaysRetrain = True
            tfDef.subsetCrossValid = True
            try:
                LearnPredictFile(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)

        if o in ("-t" , "--traintf"):
            if len(sys.argv) > 3:
                numRuns = int(sys.argv[3])
            else:
                numRuns = 1
            preprocDef.scrambleNoiseFlag = False
            try:
                TrainTF(sys.argv[2], int(numRuns))
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
            try:
                LearnPredictBatch(sys.argv[2])
            except:
                usage()
                sys.exit(2)

        if o in ("-p" , "--pca"):
            if len(sys.argv) > 3:
                numPCAcomp = int(sys.argv[3])
            else:
                numPCAcomp = pcaDef.numPCAcomponents
            try:
                runPCA(sys.argv[2], numPCAcomp)
            except:
                usage()
                sys.exit(2)

        if o in ("-k" , "--kmaps"):
            if len(sys.argv) > 3:
                numKMcomp = int(sys.argv[3])
            else:
                numKMcomp = kmDef.numKMcomponents
            try:
                KmMap(sys.argv[2], numKMcomp)
            except:
                usage()
                sys.exit(2)

#**********************************************
''' Learn and Predict - File'''
#**********************************************
def LearnPredictFile(learnFile, sampleFile):
    ''' Open and process training data '''
    En, Cl, A, YnormXind = readLearnFile(learnFile)

    learnFileRoot = os.path.splitext(learnFile)[0]

    ''' Run PCA '''
    if pcaDef.runPCA == True:
        runPCAmain(A, Cl, En)

    ''' Open prediction file '''
    R, Rx = readPredFile(sampleFile)

    ''' Preprocess prediction data '''
    A, Cl, En, Aorig = preProcessNormLearningData(A, En, Cl, YnormXind, 0)
    R, Rorig = preProcessNormPredData(R, Rx, A, En, Cl, YnormXind, 0)

    ''' Run Neural Network - sklearn'''
    if nnDef.runNN == True:
        runNN(A, Cl, R, learnFileRoot)
    
    ''' Run Neural Network - TensorFlow'''
    if dnntfDef.runDNNTF == True:
        runDNNTF(A, Cl, R, learnFileRoot)

    ''' Run Support Vector Machines '''
    if svmDef.runSVM == True:
        runSVM(A, Cl, En, R, learnFileRoot)

    ''' Tensorflow '''
    if tfDef.runTF == True:
        runTFbasic(A,Cl,R, learnFileRoot)

    ''' Plot Training Data '''
    if plotDef.createTrainingDataPlot == True:
        plotTrainData(A, En, R, plotDef.plotAllSpectra, learnFileRoot)

    ''' Run K-Means '''
    if kmDef.runKM == True:
        runKMmain(A, Cl, En, R, Aorig, Rorig)


#**********************************************
''' Process - Batch'''
#**********************************************
def LearnPredictBatch(learnFile):
    summary_filename = 'summary' + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
    makeHeaderSummary(summary_filename, learnFile)
    ''' Open and process training data '''
    En, Cl, A, YnormXind = readLearnFile(learnFile)
    A, Cl, En, Aorig = preProcessNormLearningData(A, En, Cl, YnormXind, 0)
    
    if multiproc == True:
        import multiprocessing as mp
        p = mp.Pool()
        for f in glob.glob('*.txt'):
            if (f != learnFile):
                p.apply_async(processSingleBatch, args=(f, En, Cl, A, Aorig, YnormXind, summary_filename, learnFile))
        p.close()
        p.join()
    else:
        for f in glob.glob('*.txt'):
            if (f != learnFile):
                processSingleBatch(f, En, Cl, A, Aorig, YnormXind, summary_filename, learnFile)

def processSingleBatch(f, En, Cl, A, Aorig, YnormXind, summary_filename, learnFile):
    print(' Processing file: \033[1m' + f + '\033[0m\n')
    R, Rx = readPredFile(f)
    summaryFile = [f]
    ''' Preprocess prediction data '''
    R, Rorig = preProcessNormPredData(R, Rx, A, En, Cl, YnormXind, 0)

    learnFileRoot = os.path.splitext(learnFile)[0]

    ''' Run Neural Network - sklearn'''
    if nnDef.runNN == True:
        nnPred, nnProb = runNN(A, Cl, R, learnFileRoot)
        summaryFile.extend([nnPred, nnProb])
        nnDef.alwaysRetrain = False
    
    ''' Run Neural Network - TensorFlow'''
    if dnntfDef.runDNNTF == True:
        dnntfPred, dnntfProb = runDNNTF(A, Cl, R, learnFileRoot)
        summaryFile.extend([nnPred, nnProb])
        dnntfDef.alwaysRetrain = False

    ''' Run Support Vector Machines '''
    if svmDef.runSVM == True:
        svmPred, svmProb = runSVM(A, Cl, En, R, learnFileRoot)
        summaryFile.extend([svmPred, svmProb])
        svmDef.alwaysRetrain = False

    ''' Tensorflow '''
    if tfDef.runTF == True:
        tfPred, tfProb, tfAccur = runTFbasic(A,Cl,R, learnFileRoot)
        summaryFile.extend([tfPred, tfProb, tfAccur])
        tfDef.tfalwaysRetrain = False

    ''' Run K-Means '''
    if kmDef.runKM == True:
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
    En, Cl, A, YnormXind = readLearnFile(learnFile)

    learnFileRoot = os.path.splitext(learnFile)[0]

    ''' Open prediction map '''
    X, Y, R, Rx = readPredMap(mapFile)
    type = 0
    i = 0;
    svmPred = nnPred = tfPred = kmPred = np.empty([X.shape[0]])
    A, Cl, En, Aorig = preProcessNormLearningData(A, En, Cl, YnormXind, type)
    print(' Processing map...' )
    for r in R[:]:
        r, rorig = preProcessNormPredData(r, Rx, A, En, Cl, YnormXind, type)
        type = 1

        ''' Run Neural Network - sklearn'''
        if nnDef.runNN == True:
            nnPred[i], temp = runNN(A, Cl, r, learnFileRoot)
            saveMap(mapFile, 'NN', 'HC', nnPred[i], X[i], Y[i], True)
            nnDef.alwaysRetrain = False
        
        ''' Run Neural Network - TensorFlow'''
        if nnDef.runNN == True:
            dnntfPred[i], temp = runDNNTF(A, Cl, r, learnFileRoot)
            saveMap(mapFile, 'DNN-TF', 'HC', dnntfPred[i], X[i], Y[i], True)
            dnnDef.alwaysRetrain = False
        
        ''' Run Support Vector Machines '''
        if svmDef.runSVM == True:
            svmPred[i], temp = runSVM(A, Cl, En, r, learnFileRoot)
            saveMap(mapFile, 'svm', 'HC', svmPred[i], X[i], Y[i], True)
            svmDef.alwaysRetrain = False

        ''' Tensorflow '''
        if tfDef.runTF == True:
            tfPred[i], temp, temp = runTFbasic(A,Cl,r, learnFileRoot)
            saveMap(mapFile, 'TF', 'HC', tfPred[i], X[i], Y[i], True)
            tfDef.alwaysRetrain = False

        ''' Run K-Means '''
        if kmDef.runKM == True:
            kmDef.plotKM = False
            kmPred[i] = runKMmain(A, Cl, En, r, Aorig, rorig)
            saveMap(mapFile, 'KM', 'HC', kmPred[i], X[i], Y[i], True)
        i+=1

    if nnDef.plotNN == True and nnDef.runNN == True:
        plotMaps(X, Y, nnPred, 'Deep Neural networks - sklearn')
    if nnDef.plotNN == True and nnDef.runNN == True:
        plotMaps(X, Y, dnntfPred, 'Deep Neural networks - tensorFlow')
    if svmDef.plotSVM == True and svmDef.runSVM == True:
        plotMaps(X, Y, svmPred, 'SVM')
    if tfDef.plotMapTF == True and tfDef.runTF == True:
        plotMaps(X, Y, tfPred, 'TensorFlow')
    if kmDef.plotKMmaps == True and kmDef.runKM == True:
        plotMaps(X, Y, kmPred, 'K-Means Prediction')


#********************************************************************************
''' Run Neural Network - sklearn '''
#********************************************************************************
def runNN(A, Cl, R, Root):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.externals import joblib
    
    if nnDef.MLPRegressor is False:
        Root+"/DNN-TF_"
        nnTrainedData = Root + '.nnModelC.pkl'
    else:
        nnTrainedData = Root + '.nnModelR.pkl'
    
    print('==========================================================================\n')
    print(' Running Neural Network: multi-layer perceptron (MLP)')
    print(' Number of neurons: Hidden layers:', nnDef.numNeurons)
    print(' Optimizer:',nnDef.nnOptimizer,', Activation Fn:',nnDef.activation_function)

    try:
        if nnDef.alwaysRetrain == False:
            with open(nnTrainedData):
                print(' Opening NN training model...\n')
                clf = joblib.load(nnTrainedData)
        else:
            raise ValueError('Force NN retraining.')
    except:
        #**********************************************
        ''' Retrain training data if not available'''
        #**********************************************
        if nnDef.MLPRegressor is False:
            print(' Retraining NN model using MLP Classifier...')
            clf = MLPClassifier(solver=nnDef.nnOptimizer, alpha=1e-5, activation = nnDef.activation_function,
                                hidden_layer_sizes=(nnDef.numNeurons,), random_state=1)
        else:
            print(' Retraining NN model using MLP Regressor...')
            clf = MLPRegressor(solver=nnDef.nnOptimizer, alpha=1e-5, hidden_layer_sizes=(nnDef.numNeurons,), random_state=1)
            Cl = np.array(Cl,dtype=float)

        if nnDef.subsetCrossValid == True:
            print(" Iterating training using: ",str(nnDef.percentCrossValid*100), "% as test subset, iterating",str(nnDef.iterCrossValid)," time(s) ...\n")
            for i in range(nnDef.iterCrossValid):
                As, Cls, As_cv, Cls_cv = formatSubset(A, Cl, nnDef.percentCrossValid)
                clf.fit(As, Cls)
                if nnDef.MLPRegressor is False:
                    print('  Mean accuracy: ',100*clf.score(As_cv,Cls_cv),'%')
                else:
                    print('  Coefficient of determination R^2: ',clf.score(As_cv,Cls_cv))
        else:
            print(" Training on the full training dataset\n")
            clf.fit(A, Cl)

        joblib.dump(clf, nnTrainedData)
            
    if nnDef.MLPRegressor is False:
        prob = clf.predict_proba(R)[0].tolist()
        rosterPred = np.where(clf.predict_proba(R)[0]>nnDef.thresholdProbabilityPred/100)[0]
        print('\n  ==============================')
        print('  \033[1mNN\033[0m - Probability >',str(nnDef.thresholdProbabilityPred),'%')
        print('  ==============================')
        print('  Prediction\tProbability [%]')
        for i in range(rosterPred.shape[0]):
            print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.4f}'.format(100*clf.predict_proba(R)[0][rosterPred][i])))
        print('  ==============================')

        predValue = clf.predict(R)[0]
        predProb = round(100*max(prob),4)
        print('\033[1m' + '\n Predicted classifier value (Deep Neural Networks - sklearn) = ' + str(predValue) +
          '  (probability = ' + str(predProb) + '%)\033[0m\n')
    else:
        Cl = np.array(Cl,dtype=float)
        predValue = clf.predict(R)[0]
        predProb = clf.score(A,Cl)
        print('\033[1m' + '\n Predicted regressor value (Deep Neural Networks - sklearn) = ' + str('{:.3f}'.format(predValue)) +
              '  (R^2 = ' + str('{:.5f}'.format(predProb)) + ')\033[0m\n')

    #**************************************
    ''' Neural Networks Classification Report '''
    #**************************************
    if nnDef.nnClassReport == True:
        print(' Neural Networks Classification Report\n')
        runClassReport(clf, A, Cl)

    #*************************
    ''' Plot probabilities '''
    #*************************
    if plotDef.showProbPlot == True:
        if nnDef.MLPRegressor is False:
            plotProb(clf, R)

    return predValue, predProb

#********************************************************************************
''' TensorFlow '''
''' Run SkFlow - DNN Classifier '''
''' https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier'''
#********************************************************************************
''' Run DNNClassifier model training and evaluation via TensorFlow-skflow '''
#********************************************************************************
def runDNNTF(A, Cl, R, Root):
    print('==========================================================================\n')
    print(' Running Deep Neural Networks: DNNClassifier - TensorFlow...')
    print(' Hidden layers:', dnntfDef.hidden_layers)
    print(' Optimizer:',dnntfDef.nnOptimizer,', Activation function:',dnntfDef.activation_function)
    import tensorflow as tf
    import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing
    
    if dnntfDef.alwaysRetrain == False:
        model_directory = Root + "/DNN-TF_" + str(dnntfDef.numHidlayers)+"x"+str(dnntfDef.numNeurons)
        print("\n Training model saved in: ", model_directory, "\n")
    else:
        model_directory = None
        print("\n Training model not saved\n")

    #**********************************************
    ''' Initialize Estimator and training data '''
    #**********************************************
    print(' Initializing TensorFlow...')
    tf.reset_default_graph()

    le = preprocessing.LabelEncoder()
    Cl2 = le.fit_transform(Cl)
    
    feature_columns = skflow.infer_real_valued_columns_from_input(A.astype(np.float32))
    clf = skflow.DNNClassifier(feature_columns=feature_columns, hidden_units=dnntfDef.hidden_layers,
                               optimizer=dnntfDef.nnOptimizer, n_classes=np.unique(Cl).size,
                               activation_fn=dnntfDef.activationFn, model_dir=model_directory)
                               
    print("\n Number of training steps:",dnntfDef.trainingSteps)

    #**********************************************
    ''' Train '''
    #**********************************************
    if dnntfDef.subsetCrossValid == True:
        print(" Iterating training using: ",str(dnntfDef.percentCrossValid*100), "% as test subset, iterating",str(dnntfDef.iterCrossValid)," time(s) ...\n")
        for i in range(dnntfDef.iterCrossValid):
            As, Cl2s, As_cv, Cl2s_cv = formatSubset(A, Cl2, dnntfDef.percentCrossValid)
            
            clf.fit(input_fn=lambda: input_fn(As, Cl2s), steps=dnntfDef.trainingSteps)
            accuracy_score = clf.evaluate(input_fn=lambda: input_fn(As_cv, Cl2s_cv), steps=1)
            print("\n  Accuracy: {:.2f}%".format(100*accuracy_score["accuracy"]))
            print("  Loss: {:.2f}".format(accuracy_score["loss"]))
            print("  Global step: {:.2f}\n".format(accuracy_score["global_step"]))
    else:
        print(" Training on the full training dataset\n")
        clf.fit(input_fn=lambda: input_fn(A, Cl2), steps=dnntfDef.trainingSteps)

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
    
    print('\n  ================================')
    print('  \033[1mDNN-TF\033[0m - Probability >',str(dnntfDef.thresholdProbabilityPred),'%')
    print('  ================================')
    print('  Prediction\tProbability [%]')
    for i in range(rosterPred.shape[0]):
        print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.4f}'.format(100*prob[rosterPred][i])))
    print('  ================================')
    
    print('\033[1m' + '\n Predicted regressor value (Deep Neural Networks - TensorFlow) = ' + predValue +
          '  (probability = ' + str(predProb) + '%)\033[0m\n')

    return predValue, predProb

#**********************************************
''' Format input data for Estimator '''
#**********************************************
def input_fn(A, Cl2):
    import tensorflow as tf
    x = tf.constant(A.astype(np.float32))
    y = tf.constant(Cl2)
    return x,y

#********************************************************************************
''' Run SVM '''
#********************************************************************************
def runSVM(A, Cl, En, R, Root):
    from sklearn import svm
    from sklearn.externals import joblib
    svmTrainedData = Root + '.svmModel.pkl'
    print('==========================================================================\n')
    print(' Running Support Vector Machine (kernel: ' + svmDef.kernel + ')...')
    try:
        if svmDef.alwaysRetrain == False:
            with open(svmTrainedData):
                print(' Opening SVM training model...\n')
                clf = joblib.load(svmTrainedData)
        else:
            raise ValueError('Force retraining SVM model')
    except:
        #**********************************************
        ''' Retrain training model if not available'''
        #**********************************************
        print(' Retraining SVM data...')
        clf = svm.SVC(C = svmDef.Cfactor, decision_function_shape = 'ovr', probability=True)
        
        if svmDef.subsetCrossValid == True:
            print(" Iterating training using: ",str(nnDef.percentCrossValid*100), "% as test subset, iterating",str(nnDef.iterCrossValid)," time(s) ...\n")
            for i in range(svmDef.iterCrossValid):
                As, Cls, As_cv, Cls_cv = formatSubset(A, Cl, svmDef.percentCrossValid)
                clf.fit(As, Cls)
                print('  Mean accuracy: ',100*clf.score(As_cv,Cls_cv),'%')
        else:
            print(" Training on the full training dataset\n")
            clf.fit(A,Cl)

        Z = clf.decision_function(A)
        print('\n  Number of classes = ' + str(Z.shape[1]))
        joblib.dump(clf, svmTrainedData)
        if svmDef.showClasses == True:
            print('  List of classes: ' + str(clf.classes_))

    R_pred = clf.predict(R)
    prob = clf.predict_proba(R)[0].tolist()
    
    rosterPred = np.where(clf.predict_proba(R)[0]>svmDef.thresholdProbabilitySVMPred/100)[0]
    print('\n  ==============================')
    print('  \033[1mSVM\033[0m - Probability >',str(svmDef.thresholdProbabilitySVMPred),'%')
    print('  ==============================')
    print('  Prediction\tProbability [%]')
    for i in range(rosterPred.shape[0]):
        print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.1f}'.format(100*clf.predict_proba(R)[0][rosterPred][i])))
    print('  ==============================')

    print('\033[1m' + '\n Predicted value (SVM) = ' + str(R_pred[0]) + ' (probability = ' +
      str(round(100*max(prob),1)) + '%)\033[0m\n')
    
    #**************************************
    ''' SVM Classification Report '''
    #**************************************
    if svmDef.svmClassReport == True:
        print(' SVM Classification Report \n')
        runClassReport(clf, A, Cl)

    #*************************
    ''' Plot probabilities '''
    #*************************
    if plotDef.showProbPlot == True:
        plotProb(clf, R)

    return R_pred[0], round(100*max(prob),1)

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
def runPCA(learnFile, numPCAcomponents):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from matplotlib import cm

    ''' Open and process training data '''
    En, Cl, A, YnormXind = readLearnFile(learnFile)

    print('==========================================================================\n')
    print(' Running PCA...\n')
    print(' Number of unique identifiers in training data: ' + str(np.unique(Cl).shape[0]))
    if pcaDef.customNumPCAComp == False:
        numPCAcomp = np.unique(Cl).shape[0]
    else:
        numPCAcomp = numPCAcomponents
    print(' Number of Principal components: ' + str(numPCAcomp) + '\n')
    pca = PCA(n_components=numPCAcomp)
    A_r = pca.fit(A).transform(A)

    for i in range(0,pca.components_.shape[0]):
        print(' Score PC ' + str(i) + ': ' + '{0:.0f}%'.format(pca.explained_variance_ratio_[i] * 100))
    print('')

    if plotDef.showPCAPlots == True:
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

        plt.title('Score box plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Score')
        for j in range(pca.components_.shape[0]):
            for color, i, target_name in zip(colors, range(ind.shape[0]), Cl_labels):
                plt.scatter([j+1]*len(A_r[Cl_ind==i,j]), A_r[Cl_ind==i,j], color=color, alpha=.8, lw=2, label=target_name)
        plt.boxplot(A_r)
        plt.figure()

        #******************************
        ''' Plotting Scores vs H:C '''
        #******************************
        for j in range(pca.components_.shape[0]):
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
    print('==========================================================================\n')
    print(' Running K-Means...')
    print(' Number of unique identifiers in training data: ' + str(np.unique(Cl).shape[0]))
    if kmDef.customNumKMComp == False:
        numKMcomp = np.unique(Cl).shape[0]
    else:
        numKMcomp = kmDef.numKMcomponents
    kmeans = KMeans(n_clusters=numKMcomp, random_state=0).fit(A)
    
    '''
    for i in range(0, numKMcomp):
        print('\n Class: ' + str(i) + '\n  ',end="")
        for j in range(0,kmeans.labels_.shape[0]):
            if kmeans.labels_[j] == i:
                print(' ' + str(Cl[j]), end="")
    '''
    print('\n  ==============================')
    print('  \033[1mKM\033[0m - Predicted class: \033[1m',str(kmeans.predict(R)[0]),'\033[0m')
    print('  ==============================')
    print('  Prediction')
    for j in range(0,kmeans.labels_.shape[0]):
        if kmeans.labels_[j] == 22:
            print('  ' + str(Cl[j]))
    print('  ==============================\n')


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
    kmeans = KMeans(n_clusters=kmDef.numKMcomponents, random_state=0).fit(R)
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
    Atemp = A[:,range(len(preprocDef.enSel))]

    if preprocDef.cherryPickEnPoint == True and preprocDef.enRestrictRegion == False:
        enPoints = list(preprocDef.enSel)
        enRange = list(preprocDef.enSel)

        for i in range(0, len(preprocDef.enSel)):
            enRange[i] = np.where((En<float(preprocDef.enSel[i]+preprocDef.enSelDelta[i])) & (En>float(preprocDef.enSel[i]-preprocDef.enSelDelta[i])))[0].tolist()

            for j in range(0, A.shape[0]):
                Atemp[j,i] = A[j,A[j,enRange[i]].tolist().index(max(A[j, enRange[i]].tolist()))+enRange[i][0]]

            enPoints[i] = int(np.average(enRange[i]))
        A = Atemp
        En = En[enPoints]

        if type == 0:
            print( ' Cheery picking points in the spectra\n')

    # Find index corresponding to energy value to be used for Y normalization
    if preprocDef.fullYnorm == True:
        YnormXind = np.where(En>0)[0].tolist()
    else:
        YnormXind_temp = np.where((En<float(preprocDef.YnormX+preprocDef.YnormXdelta)) & (En>float(preprocDef.YnormX-preprocDef.YnormXdelta)))[0].tolist()
        if YnormXind_temp == []:
            print( ' Renormalization region out of requested range. Normalizing over full range...\n')
            YnormXind = np.where(En>0)[0].tolist()
        else:
            YnormXind = YnormXind_temp

    print(' Number of datapoints = ' + str(A.shape[0]))
    print(' Size of each datapoint = ' + str(A.shape[1]) + '\n')
    return En, Cl, A, YnormXind

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

    if preprocDef.cherryPickEnPoint == True and preprocDef.enRestrictRegion == False:
        Rtemp = R[range(len(preprocDef.enSel))]
        enPoints = list(preprocDef.enSel)
        enRange = list(preprocDef.enSel)
        for i in range(0, len(preprocDef.enSel)):
            enRange[i] = np.where((Rx<float(preprocDef.enSel[i]+preprocDef.enSelDelta[i])) & (Rx>float(preprocDef.enSel[i]-preprocDef.enSelDelta[i])))[0].tolist()
            Rtemp[i] = R[R[enRange[i]].tolist().index(max(R[enRange[i]].tolist()))+enRange[i][0]]

            enPoints[i] = int(np.average(enRange[i]))
        R = Rtemp
        Rx = Rx[enPoints]

    return R, Rx

#**********************************************************************************
''' Preprocess Learning data '''
#**********************************************************************************
def preProcessNormLearningData(A, En, Cl, YnormXind, type):
    print(' Processing Training data file... ')
    #**********************************************************************************
    ''' Reformat x-axis in case it does not match that of the training data '''
    #**********************************************************************************
    if preprocDef.scrambleNoiseFlag == True:
        print(' Adding random noise to training set \n')
        scrambleNoise(A, preprocDef.scrambleNoiseOffset)
    Aorig = np.copy(A)

    #**********************************************
    ''' Normalize/preprocess if flags are set '''
    #**********************************************
    if preprocDef.Ynorm == True:
        if type == 0:
            if preprocDef.fullYnorm == False:
                print('  Normalizing spectral intensity to: ' + str(preprocDef.YnormTo) + '; En = [' + str(preprocDef.YnormX-preprocDef.YnormXdelta) + ', ' + str(preprocDef.YnormX+preprocDef.YnormXdelta) + ']')
            else:
                print('  Normalizing spectral intensity to: ' + str(preprocDef.YnormTo) + '; to max intensity in spectra')
        for i in range(0,A.shape[0]):
            if(np.amin(A[i]) <= 0):
                A[i,:] = A[i,:] - np.amin(A[i,:]) + 0.00001
            A[i,:] = np.multiply(A[i,:], preprocDef.YnormTo/A[i,A[i][YnormXind].tolist().index(max(A[i][YnormXind].tolist()))+YnormXind[0]])

    if preprocDef.StandardScalerFlag == True:
        print('  Using StandardScaler from sklearn ')
        A = preprocDef.scaler.fit_transform(A)

    #**********************************************
    ''' Energy normalization range '''
    #**********************************************
    if preprocDef.enRestrictRegion == True:
        A = A[:,range(preprocDef.enLim1, preprocDef.enLim2)]
        En = En[range(preprocDef.enLim1, preprocDef.enLim2)]
        Aorig = Aorig[:,range(preprocDef.enLim1, preprocDef.enLim2)]

        if type == 0:
            print( '  Restricting energy range between: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    else:
        if type == 0:
            if(preprocDef.cherryPickEnPoint == True):
                print( '  Using selected spectral points:')
                print(En)
            else:
                print( '  Using full energy range: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')

    return A, Cl, En, Aorig

#**********************************************************************************
''' Preprocess Prediction data '''
#**********************************************************************************
def preProcessNormPredData(R, Rx, A, En, Cl, YnormXind, type):
    print(' Processing Prediction data file... ')
    #**********************************************************************************
    ''' Reformat x-axis in case it does not match that of the training data '''
    #**********************************************************************************
    if(R.shape[0] != A.shape[1]):
        if type == 0:
            print('\033[1m' + '  WARNING: Different number of datapoints for the x-axis\n  for training (' + str(A.shape[1]) + ') and sample (' + str(R.shape[0]) + ') data.\n  Reformatting x-axis of sample data...\n' + '\033[0m')
        R = np.interp(En, Rx, R)
    R = R.reshape(1,-1)
    Rorig = np.copy(R)

    #**********************************************
    ''' Normalize/preprocess if flags are set '''
    #**********************************************
    if preprocDef.Ynorm == True:
        if type == 0:
            if preprocDef.fullYnorm == False:
                print('  Normalizing spectral intensity to: ' + str(preprocDef.YnormTo) + '; En = [' + str(preprocDef.YnormX-preprocDef.YnormXdelta) + ', ' + str(preprocDef.YnormX+preprocDef.YnormXdelta) + ']')
            else:
                print('  Normalizing spectral intensity to: ' + str(preprocDef.YnormTo) + '; to max intensity in spectra')
    
        if(np.amin(R) <= 0):
            print('  Spectra max below zero detected')
            R[0,:] = R[0,:] - np.amin(R[0,:]) + 0.00001
        R[0,:] = np.multiply(R[0,:], preprocDef.YnormTo/R[0,R[0][YnormXind].tolist().index(max(R[0][YnormXind].tolist()))+YnormXind[0]])

    if preprocDef.StandardScalerFlag == True:
        print('  Using StandardScaler from sklearn ')
        R = preprocDef.scaler.transform(R)
    
    #**********************************************
    ''' Energy normalization range '''
    #**********************************************
    if preprocDef.enRestrictRegion == True:
        A = A[:,range(preprocDef.enLim1, preprocDef.enLim2)]
        En = En[range(preprocDef.enLim1, preprocDef.enLim2)]
        R = R[:,range(preprocDef.enLim1, preprocDef.enLim2)]
        
        if type == 0:
            print( '  Restricting energy range between: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    else:
        if type == 0:
            if(preprocDef.cherryPickEnPoint == True):
                print( '  Using selected spectral points:')
                print(En)
            else:
                print( '  Using full energy range: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    return R, Rorig

#**********************************************************************************
''' Preprocess prediction data '''
#**********************************************************************************
def preProcessNormMap(A, En, type):
    #**********************************************************************************
    ''' Reformat x-axis in case it does not match that of the training data '''
    #**********************************************************************************

    # Find index corresponding to energy value to be used for Y normalization
    if preprocDef.fullYnorm == False:
        YnormXind = np.where((En<float(preprocDef.YnormX+preprocDef.YnormXdelta)) & (En>float(preprocDef.YnormX-preprocDef.YnormXdelta)))[0].tolist()
    else:
        YnormXind = np.where(En>0)[0].tolist()
    Aorig = np.copy(A)

    #**********************************************
    ''' Normalize/preprocess if flags are set '''
    #**********************************************
    if preprocDef.Ynorm == True:
        if type == 0:
            print(' Normalizing spectral intensity to: ' + str(preprocDef.YnormTo) + '; En = [' + str(preprocDef.YnormX-preprocDef.YnormXdelta) + ', ' + str(preprocDef.YnormX+preprocDef.YnormXdelta) + ']')
        for i in range(0,A.shape[0]):
            A[i,:] = np.multiply(A[i,:], preprocDef.YnormTo/np.amax(A[i]))


    if preprocDef.StandardScalerFlag == True:
        print('  Using StandardScaler from sklearn ')
        A = preprocDef.scaler.fit_transform(A)

    #**********************************************
    ''' Energy normalization range '''
    #**********************************************
    if preprocDef.enRestrictRegion == True:
        A = A[:,range(preprocDef.enLim1, preprocDef.enLim2)]
        En = En[range(preprocDef.enLim1, preprocDef.enLim2)]
        Aorig = Aorig[:,range(preprocDef.enLim1, preprocDef.enLim2)]
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
    return A_train, Cl_train, A_cv, Cl_cv

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
def plotTrainData(A, En, R, plotAllSpectra, learnFileRoot):
    import matplotlib.pyplot as plt
    if plotDef.plotAllSpectra == True:
        step = 1
        learnFileRoot = learnFileRoot + '_full-set'
    else:
        step = plotDef.stepSpectraPlot
        learnFileRoot = learnFileRoot + '_partial-' + str(step)

    print(' Plotting Training dataset in: ' + learnFileRoot + '.png\n')
    if preprocDef.Ynorm ==True:
        plt.title('Normalized Training Data')
    else:
        plt.title('Training Data')
    for i in range(0,A.shape[0], step):
        plt.plot(En, A[i,:], label='Training data')
    plt.plot(En, R[0,:], linewidth = 4, label='Sample data')
    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')

    plt.savefig(learnFileRoot + '.png', dpi = 160, format = 'png')  # Save plot

    if plotDef.showTrainingDataPlot == True:
        plt.show()

    plt.close()


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
        summaryHeader2 = ['File','SVM-HC','SVM-Prob%', 'NN-HC', 'NN-Prob%', 'TF-HC', 'TF-Prob%', 'TF-Accuracy%']
        with open(file, "a") as sum_file:
            csv_out=csv.writer(sum_file)
            csv_out.writerow(summaryHeader1)
            csv_out.writerow(summaryHeader2)
            sum_file.close()

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print(' Single files:')
    print('  python3 SpectraLearnPredict.py -f <learningfile> <spectrafile> \n')
    print(' Single files with cross-validation for accuracy determination: ')
    print('  python3 SpectraLearnPredict.py -a <learningfile> <spectrafile> \n')
    print(' Maps (formatted for Horiba LabSpec):')
    print('  python3 SpectraLearnPredict.py -m <learningfile> <spectramap> \n')
    print(' Batch txt files:')
    print('  python3 SpectraLearnPredict.py -b <learningfile> \n')
    print(' K-means on maps:')
    print('  python3 SpectraLearnPredict.py -k <spectramap> <number_of_classes>\n')
    print(' Principal component analysis on spectral collection files: ')
    print('  python3 SpectraLearnPredict.py -p <spectrafile> <#comp>\n')
    print(' Run tensorflow training only:')
    print('  python3 SpectraLearnPredict.py -t <learningfile> <# iterations>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')


#************************************
''' Info on Classification Report '''
#************************************
def runClassReport(clf, A, Cl):
    from sklearn.metrics import classification_report
    y_pred = clf.predict(A)
    print(classification_report(Cl, y_pred, target_names=clf.classes_))
    print(' Precision is the probability that, given a classification result for a sample,\n' +
          ' the sample actually belongs to that class. Recall (Accuracy) is the probability that a \n' +
          ' sample will be correctly classified for a given class. f1-score combines both \n' +
          ' accuracy and precision to give a single measure of relevancy of the classifier results.\n')

#************************************
''' Introduce Noise in Data '''
#************************************
def scrambleNoise(A, offset):
    from random import uniform
    for i in range(A.shape[1]):
        A[:,i] += offset*uniform(-1,1)


#********************************************************************************
''' Tensorflow '''
''' https://www.tensorflow.org/get_started/mnist/beginners'''
#********************************************************************************
''' Setup training-only via TensorFlow '''
#********************************************************************************
def TrainTF(learnFile, numRuns):
    learnFileRoot = os.path.splitext(learnFile)[0]
    summary_filename = learnFileRoot + '_summary-TF-training' + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.log'))
    tfDef.alwaysRetrain = True
    tfDef.alwaysImprove = True
    
    ''' Open and process training data '''
    En, Cl, A, YnormXind = readLearnFile(learnFile)
    En_temp = En
    Cl_temp = Cl
    A_temp = A
    with open(summary_filename, "a") as sum_file:
        sum_file.write(str(datetime.now().strftime('Training started: %Y-%m-%d %H:%M:%S\n')))
        if preprocDef.scrambleNoiseFlag == True:
            sum_file.write(' Using Noise scrambler (offset: ' + str(preprocDef.scrambleNoiseOffset) + ')\n\n')
        sum_file.write('\nIteration\tAccuracy %\t Prediction\t Probability %\n')

    index = random.randint(0,A.shape[0]-1)
    R = A[index,:]

    if preprocDef.scrambleNoiseFlag == False:
        A_temp, Cl_temp, En_temp, Aorig = preProcessNormLearningData(A, En, Cl, YnormXind, 0)
        ''' Plot Training Data '''
        if plotDef.createTrainingDataPlot == True:
            plotTrainData(A, En, R.reshape(1,-1), plotDef.plotAllSpectra, learnFileRoot)

    for i in range(numRuns):
        print(' Running tensorflow training iteration: ' + str(i+1) + '\n')
        ''' Preprocess prediction data '''
        if preprocDef.scrambleNoiseFlag == True:
            A_temp, Cl_temp, En_temp, Aorig = preProcessNormLearningData(A, En, Cl, YnormXind, 0)
        R_temp, Rorig = preProcessNormPredData(R, En, A_temp, En_temp, Cl_temp, YnormXind, 0)
        print(' Using random spectra from training dataset as evaluation file ')
        tfPred, tfProb, tfAccur = runTensorFlow(A_temp,Cl_temp,R_temp,learnFileRoot)
        
        with open(summary_filename, "a") as sum_file:
            sum_file.write(str(i+1) + '\t{:10.2f}\t'.format(tfAccur) + str(tfPred) + '\t{:10.2f}\n'.format(tfProb))

        print(' Nominal class for prediction spectra:', str(index+1), '\n')
    
    with open(summary_filename, "a") as sum_file:
        sum_file.write(str(datetime.now().strftime('\nTraining ended: %Y-%m-%d %H:%M:%S\n')))
    
    print(' Completed ' + str(numRuns) + ' Training iterations. \n')

#********************************************************************************
''' Format vectors of unique labels '''
#********************************************************************************
def formatClass(rootFile, Cl):
    import sklearn.preprocessing as pp
    print('==========================================================================\n')
    print(' Running basic TensorFlow. Creating class data in binary form...')
    Cl2 = pp.LabelBinarizer().fit_transform(Cl)
    
    import matplotlib.pyplot as plt
    plt.hist([float(x) for x in Cl], bins=np.unique([float(x) for x in Cl]), edgecolor="black")
    plt.xlabel('Class')
    plt.ylabel('Occurrances')
    plt.title('Class distibution')
    plt.savefig(rootFile + '_ClassDistrib.png', dpi = 160, format = 'png')  # Save plot
    if tfDef.plotClassDistribTF == True:
        print(' Plotting Class distibution \n')
        plt.show()
    
    return Cl2

#********************************************************************************
''' Run basic model training and evaluation via TensorFlow '''
#********************************************************************************
def runTFbasic(A, Cl, R, Root):
    import tensorflow as tf
    tfTrainedData = Root + '.tfmodel'
    Cl2 = formatClass(Root, Cl)
    
    print(' Initializing TensorFlow...')
    tf.reset_default_graph()
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
    
    if tfDef.decayLearnRate == True:
        print(' Using decaying learning rate, start at:',tfDef.learnRate, '\n')
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = tfDef.learnRate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    else:
        print(' Using fix learning rate:', tfDef.learnRate, '\n')
        train_step = tf.train.GradientDescentOptimizer(tfDef.learnRate).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    if tfDef.enableTensorboard == True:
        writer = tf.summary.FileWriter(".", sess.graph)
        print('\n Saving graph. Accessible via tensorboard.  \n')
    
    saver = tf.train.Saver()
    accur = 0
    try:
        if tfDef.alwaysRetrain == False:
            print(' Opening TF training model from:', tfTrainedData)
            saver.restore(sess, './' + tfTrainedData)
            print('\n Model restored.\n')
        else:
            raise ValueError(' Force TF model retraining.')
    except:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        if os.path.isfile(tfTrainedData + '.meta') & tfDef.alwaysImprove == True:
            print('\n Improving TF model...')
            saver.restore(sess, './' + tfTrainedData)
        else:
            print('\n Rebuildind TF model...')
        
        if tfDef.subsetCrossValid == True:
            print(' Iterating training using subset (' +  str(tfDef.percentCrossValid*100) + '%), ' + str(tfDef.iterCrossValid) + ' times ...')
            for i in range(tfDef.iterCrossValid):
                As, Cl2s, As_cv, Cl2s_cv = formatSubset(A, Cl2, tfDef.percentCrossValid)
                summary = sess.run(train_step, feed_dict={x: As, y_: Cl2s})
                correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                accur = 100*accuracy.eval(feed_dict={x:As_cv, y_:Cl2s_cv})
        else:
            summary = sess.run(train_step, feed_dict={x: A, y_: Cl2})
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accur = 100*accuracy.eval(feed_dict={x:A, y_:Cl2})

        save_path = saver.save(sess, tfTrainedData)
        print(' Model saved in file: %s\n' % save_path)
        print('\033[1m Accuracy: ' + str('{:.3f}'.format(accur)) + '%\n\033[0m')
    
    if tfDef.enableTensorboard == True:
        writer.close()

    res1 = sess.run(y, feed_dict={x: R})
    res2 = sess.run(tf.argmax(y, 1), feed_dict={x: R})
    
    sess.close()
    
    rosterPred = np.where(res1[0]>tfDef.thresholdProbabilityTFPred)[0]
    print('  ==============================')
    print('  \033[1mTF\033[0m - Probability >',str(tfDef.thresholdProbabilityTFPred),'%')
    print('  ==============================')
    print('  Prediction\tProbability [%]')
    for i in range(rosterPred.shape[0]):
        print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.1f}'.format(res1[0][rosterPred][i])))
    print('  ==============================\n')
    
    print('\033[1m Predicted value (TF): ' + str(np.unique(Cl)[res2][0]) + ' (Probability: ' + str('{:.1f}'.format(res1[0][res2][0])) + '%)\n' + '\033[0m' )
    return np.unique(Cl)[res2][0], res1[0][res2][0], accur

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
