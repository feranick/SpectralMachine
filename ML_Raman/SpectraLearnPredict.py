#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
*********************************************
*
* SpectraLearnPredict
* Perform Machine Mearning on Raman data.
* version: 20161010a
*
* Uses: PCA, SVM, Neural Networks, TensorFlow
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
**********************************************
'''
print(__doc__)

import matplotlib
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Agg')

import numpy as np
import sys, os.path, getopt, glob, csv
from os.path import exists
from os import rename

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

svmTrainedData = "svmTrained.pkl"
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

nnTrainedData = "nnTrained.pkl"
nnAlwaysRetrain = True

''' Solver for NN
    lbfgs preferred for small datasets
    (alternatives: 'adam' or 'sgd') '''
nnSolver = 'lbfgs'
nnNeurons = 100  #default = 100

#**********************************************
''' Principal component analysis (PCA) '''
#**********************************************
runPCA = False
numPCAcomp = 5

#**********************************************
''' TensorFlow '''
#**********************************************
runTF = False

#**********************************************
''' Plotting '''
#**********************************************
showProbPlot = False
showTrainingDataPlot = False

#**********************************************
''' Main '''
#**********************************************
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "fmh:", ["file", "map", "help"])
    except:
        usage()
        sys.exit(2)

    if opts == []:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-f" , "--file"):
            try:
                LearnPredict(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)
        if o in ("-m" , "--map"):
            try:
                print(' NOT IMPLEMENTED YET')
            except:
                usage()
                sys.exit(2)

#**********************************************
''' Learn and Predict '''
#**********************************************
def LearnPredict(mapFile, sampleFile):
    
    #**********************************************
    ''' Open and process training data '''
    #**********************************************
    try:
        with open(mapFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Map data file not found \n' + '\033[0m')
        return

    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(M,np.s_[0:1],0)
    Cl = ['{:.2f}'.format(x) for x in M[:,0]]
    A = np.delete(M,np.s_[0:1],1)
    AmaxIndex = A.shape[1]

    # Find index corresponding to energy value to be used for Y normalization
    if fullYnorm == False:
        YnormXind = np.where((En<float(YnormX+YnormXdelta)) & (En>float(YnormX-YnormXdelta)))[0].tolist()
    else:
        YnormXind = np.where(En>0)[0].tolist()

    Amax = np.empty([A.shape[0],1])
    print(' Number of datapoints = ' + str(A.shape[0]))
    print(' Size of each datapoint = ' + str(A.shape[1]) + '\n')

    #**********************************************
    ''' Open prediction file '''
    #**********************************************
    try:
        with open(sampleFile, 'r') as f:
            print(' Opening sample data for prediction...')
            Rtot = np.loadtxt(f, unpack =True)
            R=Rtot[1,:]
            Rx=Rtot[0,:]
    except:
        print('\033[1m' + '\n Sample data file not found \n ' + '\033[0m')
        return
    
    #**********************************************************************************
    ''' Reformat x-axis in case it does not match that of the training data '''
    #**********************************************************************************
    if(R.shape[0] != AmaxIndex):
        print('\033[1m' + '\n WARNING: Different number of datapoints for the x-axis\n for training (' + str(AmaxIndex) + ') and sample (' + str(R.shape[0]) + ') data.\n Reformatting x-axis of sample data...\n' + '\033[0m')
        R = np.interp(En, Rx, R)
    R = R.reshape(1,-1)

    #**********************************************
    ''' Normalize/preprocess if flags are set '''
    #**********************************************
    if Ynorm == True:
        print(' Normalizing spectral intensity to: ' + str(YnormTo) + '; En = [' + str(YnormX-YnormXdelta) + ', ' + str(YnormX+YnormXdelta) + ']\n')
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
        print( ' Restricting energy range between: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']')
    else:
        print( ' Using full energy range: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']')

    #***********************************
    ''' Run Support Vector Machines '''
    #***********************************
    if runSVM == True:
        runSVMmain(A, Cl, En, R)

    #********************
    ''' Run PCA '''
    #********************
    if runPCA == True:
        runPCAmain(En, A)
    
    #******************************
    ''' Run Neural Network '''
    #******************************
    if runNN == True:
        runNNmain(A, Cl, R)

    #********************
    ''' Tensorflow '''
    #********************
    if runTF == True:
        runTensorFlow(A,Cl,R)

    #***************************
    ''' Plot Training Data '''
    #***************************
    if showTrainingDataPlot == True:
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

#********************
''' Run SVM '''
#********************
def runSVMmain(A, Cl, En, R):
    from sklearn import svm
    from sklearn.externals import joblib
    print('\n Running Support Vector Machine (kernel: ' + kernel + ')...')
    try:
        if svmAlwaysRetrain == False:
            with open(svmTrainedData):
                print(' Opening SVM training data...')
                clf = joblib.load(svmTrainedData)
        else:
            raise ValueError('Force retraining SVM data')
    except:
        #**********************************************
        ''' Retrain NN data if not available'''
        #**********************************************
        print('\n Retraining SVM data...')
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



#*************************
''' Run Neural Network '''
#*************************
def runNNmain(A, Cl, R):
    from sklearn.neural_network import MLPClassifier
    from sklearn.externals import joblib
    print('\n Running Neural Network: multi-layer perceptron (MLP) - (solver: ' + nnSolver + ')...')
    try:
        if nnAlwaysRetrain == False:
            with open(nnTrainedData):
                print(' Opening NN training data...')
                clf = joblib.load(nnTrainedData)
        else:
            raise ValueError('Force NN retraining.')
    except:
        #**********************************************
        ''' Retrain data if not available'''
        #**********************************************
        print('\n Retraining NN data...')
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

#********************
''' Run PCA '''
#********************
def runPCAmain(En, A):
    from sklearn.decomposition import PCA, RandomizedPCA
    import matplotlib.pyplot as plt
    print(' Running PCA...\n')
    pca = PCA(n_components=numPCAcomp)
    pca.fit_transform(A)
    for i in range(0,pca.components_.shape[0]):
        plt.plot(En, pca.components_[i,:], label=i)
    #plt.plot(En, pca.components_[1,:]-pca.components_[0,:], label='Difference')
    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('PCA')
    plt.legend()
    plt.show()


#********************************************************************************
''' Tensorflow (Experimental)'''
''' https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html'''
#********************************************************************************
def runTensorFlow(A, Cl, R):
    import tensorflow as tf
    formatClassfile = "tfFormatClass.txt"
    
    try:
        with open(formatClassfile) as f:
            print(' Opening TensorFlow format class data...\n')
            Cl2 = np.loadtxt(f, unpack =False)
    except:
        print( ' Formatting training cluster data...\n')
        Cl2 = np.zeros((np.array(Cl).shape[0], np.unique(Cl).shape[0]))
        
        for i in range(np.array(Cl).shape[0]):
            for j in range(np.array(np.unique(Cl).shape[0])):
                if np.array(Cl)[i] == np.unique(Cl)[j]:
                    np.put(Cl2, [i, j], 1)
                else:
                    np.put(Cl2, [i, j], 0)
                
                np.savetxt(formatClassfile, Cl2, delimiter='\t', fmt='%10.6f')

    print(' Initializing TensorFlow...\n')
    x = tf.placeholder(tf.float32, [None, A.shape[1]])
    W = tf.Variable(tf.zeros([A.shape[1], np.unique(Cl).shape[0]]))
    b = tf.Variable(tf.zeros(np.unique(Cl).shape[0]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    print(' Training TensorFlow...\n')
    y_ = tf.placeholder(tf.float32, [None, np.unique(Cl).shape[0]])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    sess.run(train_step, feed_dict={x: A, y_: Cl2})
    res1 = sess.run(y, feed_dict={x: R})
    res2 = sess.run(tf.argmax(y, 1), feed_dict={x: R})
    #print(sess.run(accuracy, feed_dict={x: R, y_: Cl2}))
    print('\033[1m' + ' Prediction (TF): ' + str(np.unique(Cl)[res2][0]) + ' (' + str('{:.1f}'.format(res1[0][res2][0]*100)) + '%)\n' + '\033[0m' )

#************************************
''' Plot Probabilities '''
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


####################################################################
''' Read map files '''
####################################################################
def readMap(mapFile):
    try:
        with open(mapFile, 'r') as f:
            En = np.array(f.readline().split(), dtype=np.dtype(float))
            A = np.loadtxt(f, unpack =False)
        A = np.delete(A, np.s_[0:2], 1)
        print(' Shape map: ' + str(A.shape))
        return A, En
    except:
        print('\033[1m' + ' Map data file not found \n' + '\033[0m')
        return

####################################################################
''' Save map files '''
####################################################################
def saveMap(file, out, extension, s, x1, y1):
    inputFile = os.path.splitext(file)[0] + '_' + extension + '_map.csv'
    with open(inputFile, "a") as coord_file:
        coord_file.write('{:},'.format(x1))
        coord_file.write('{:},'.format(y1))
        if (out.success == True and out.redchi < defPar.redchi):
            coord_file.write('{:}\n'.format(s))
        else:
            coord_file.write('{:}\n'.format(defPar.outliar))
        coord_file.close()

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:')
    print(' single files: ')
    print('  python SpectraLearnPredictSVM.py -f <learningfile> <spectrafile> \n')
    print(' maps: ')
    print('  python SpectraLearnPredictSVM.py -m <learningfile> <spectramap> \n')

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
