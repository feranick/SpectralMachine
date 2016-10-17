#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict
* Perform Machine Mearning on Raman data/maps.
* version: 20161017b
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
from os.path import exists
from os import rename
from datetime import datetime, date

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

svmTrainedData = "svmModel.pkl"
class svmDef:
    svmAlwaysRetrain = True

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

nnTrainedData = "nnModel.pkl"
class nnDef:
    nnAlwaysRetrain = True

''' Solver for NN
    lbfgs preferred for small datasets
    (alternatives: 'adam' or 'sgd') '''
nnSolver = 'lbfgs'
nnNeurons = 100  #default = 100

#**********************************************
''' TensorFlow '''
#**********************************************
runTF = True

tfTrainedData = "tfmodel.ckpt"
class tfDef:
    tfAlwaysRetrain = True

#**********************************************
''' Principal component analysis (PCA) '''
#**********************************************
runPCA = False
customNumPCAComp = False
numPCAcomponents = 5

#**********************************************
''' K-means '''
#**********************************************
runKM = False
customNumKMComp = False
numKMcomponents = 20
plotKM = True

#**********************************************
''' Plotting '''
#**********************************************
showProbPlot = False
showTrainingDataPlot = False

#**********************************************
''' Multiprocessing '''
#**********************************************
multiproc = False

#**********************************************
''' Main '''
#**********************************************
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "fmbh:", ["file", "map", "batch", "help"])
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

#**********************************************
''' Learn and Predict - File'''
#**********************************************
def LearnPredictFile(learnFile, sampleFile):
    
    ''' Open and process training data '''
    En, Cl, A, Amax, YnormXind = readLearnFile(learnFile)

    ''' Open prediction file '''
    R, Rx = readPredFile(sampleFile)
    
    ''' Preprocess prediction data '''
    A, Cl, En, R, Aorig, Rorig = preProcessNormData(R, Rx, A, En, Cl, Amax, YnormXind, 0)
    
    ''' Run Support Vector Machines '''
    if runSVM == True:
        runSVMmain(A, Cl, En, R)

    ''' Run Neural Network '''
    if runNN == True:
        runNNmain(A, Cl, R)

    ''' Tensorflow '''
    if runTF == True:
        runTensorFlow(A,Cl,R)

    ''' Plot Training Data '''
    if showTrainingDataPlot == True:
        plotTrainData(A, En, R)

    ''' Run PCA '''
    if runPCA == True:
        runPCAmain(A, Cl, En, R)

    ''' Run K-Means '''
    if runKM == True:
        runKMmain(A, Cl, En, R, Aorig, Rorig)


#**********************************************
''' Process - Batch'''
#**********************************************

def processSingleBatch(f, En, Cl, A, Amax, YnormXind, summary_filename):
    print(f)
    R, Rx = readPredFile(f)
    summaryFile = [f]
    ''' Preprocess prediction data '''
    A, Cl, En, R, Aorig = preProcessNormData(R, Rx, A, En, Cl, Amax, YnormXind, 0)
            
    ''' Run Support Vector Machines '''
    if runSVM == True:
        svmPred, svmProb = runSVMmain(A, Cl, En, R)
        summaryFile.extend([svmPred, svmProb])
        svmDef.svmAlwaysRetrain = False
            
    ''' Run Neural Network '''
    if runNN == True:
        nnPred, nnProb = runNNmain(A, Cl, R)
        summaryFile.extend([nnPred, nnProb])
        nnDef.nnAlwaysRetrain = False
            
    ''' Tensorflow '''
    if runTF == True:
        tfPred, tfProb = runTensorFlow(A,Cl,R)
        summaryFile.extend([tfPred, tfProb])
            
    with open(summary_filename, "a") as sum_file:
        csv_out=csv.writer(sum_file)
        csv_out.writerow(summaryFile)
        sum_file.close()


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
                p.apply_async(processSingleBatch, args=(f, En, Cl, A, Amax, YnormXind, summary_filename))
        p.close()
        p.join()
    else:
        for f in glob.glob('*.txt'):
            if (f != learnFile):
                processSingleBatch(f, En, Cl, A, Amax, YnormXind, summary_filename)


#**********************************************
''' Learn and Predict - Maps'''
#**********************************************
def LearnPredictMap(learnFile, mapFile):
    
    ''' Open and process training data '''
    En, Cl, A, Amax, YnormXind = readLearnFile(learnFile)

    ''' Open prediction map '''
    X, Y, R, Rx = readPredMap(mapFile)
    type = 0
    i = 0;
    for r in R[:]:
        A, Cl, En, r, Aorig = preProcessNormData(r, Rx, A, En, Cl, Amax, YnormXind, type)
        type = 1
        print('\n Preprocessing map\n')

        ''' Run Support Vector Machines '''
        if runSVM == True:
            svmPred = runSVMmain(A, Cl, En, r)
            saveMap(mapFile, 'svm', 'HC', svmPred, X[i], Y[i])
            svmDef.svmAlwaysRetrain = False
    
        ''' Run Neural Network '''
        if runNN == True:
            nnPred = runNNmain(A, Cl, r)
            saveMap(mapFile, 'NN', 'HC', nnPred, X[i], Y[i])
            nnDef.nnAlwaysRetrain = False
    
        ''' Tensorflow '''
        if runTF == True:
            tfPred = runTensorFlow(A,Cl,r)
            saveMap(mapFile, 'TF', 'HC', tfPred, X[i], Y[i])
        i = i+1

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
                print(' Model restored. \n')
        else:
            raise ValueError(' Force TF model retraining.')
    except:
        print(' Retraining TF model...')
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init)
        sess.run(train_step, feed_dict={x: A, y_: Cl2})

        save_path = saver.save(sess, tfTrainedData)
        print(' Model saved in file: %s\n' % save_path)

    res1 = sess.run(y, feed_dict={x: R})
    res2 = sess.run(tf.argmax(y, 1), feed_dict={x: R})
    
    #print(sess.run(accuracy, feed_dict={x: R, y_: Cl2}))
    print('\033[1m' + ' Predicted value (TF): ' + str(np.unique(Cl)[res2][0]) + ' (' + str('{:.1f}'.format(res1[0][res2][0]*100)) + '%)\n' + '\033[0m' )
    return np.unique(Cl)[res2][0], res1[0][res2][0]*100


#********************
''' Run PCA '''
#********************
def runPCAmain(A, Cl, En, R):
    from sklearn.decomposition import PCA, RandomizedPCA
    import matplotlib.pyplot as plt
    print(' Running PCA...\n')
    print(' Number of unique identifiers in training data: ' + str(np.unique(Cl).shape[0]))
    if customNumPCAComp == False:
        numPCAcomp = np.unique(Cl).shape[0]
    else:
        numPCAcomp = numPCAcomponents
    pca = PCA(n_components=numPCAcomp)
    pca.fit_transform(A)
    for i in range(0,pca.components_.shape[0]):
        plt.plot(En, pca.components_[i,:])
    #plt.plot(En, R[0,:], linewidth = 2, label='Predict')
    #plt.plot(En, pca.components_[1,:]-pca.components_[0,:], label='Difference')
    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('PCA')
    plt.legend()
    plt.show()


#********************
''' Run K-Means '''
#********************
def runKMmain(A, Cl, En, R, Aorig, Rorig):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
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
    if plotKM == True:
        for j in range(0,kmeans.labels_.shape[0]):
            if kmeans.labels_[j] == kmeans.predict(R)[0]:
                plt.plot(En, Aorig[j,:])
        plt.plot(En, Rorig[0,:], linewidth = 2, label='Predict')
        plt.title('K-Means')
        plt.xlabel('Raman shift [1/cm]')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()


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
    plt.plot(En, R[0,:], linewidth = 2, label='Sample data')
    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')
    plt.show()

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
    Aorig = A
    Rorig = R

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
        from sklearn.model_selection import train_test_split
        if type == 0:
            print(' Selecting subset (' +  str(percentCrossValid*100) + '%) of training data for cross validation...\n')
        A_train, A_cv, Cl_train, Cl_cv = \
        train_test_split(A, Cl, test_size=percentCrossValid, random_state=42)
        A=A_train
        Cl=Cl_train

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
            print( ' Using full energy range: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')

    return A, Cl, En, R, Aorig, Rorig


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
    #AmaxIndex = A.shape[1]
    
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
    return R, Rx

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
def saveMap(file, type, extension, s, x1, y1):
    inputFile = os.path.splitext(file)[0] + '_' + type + '-' + extension + '_map.csv'
    with open(inputFile, "a") as coord_file:
        coord_file.write('{:},'.format(x1))
        coord_file.write('{:},'.format(y1))
        coord_file.write('{:}\n'.format(s))
        coord_file.close()

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
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
