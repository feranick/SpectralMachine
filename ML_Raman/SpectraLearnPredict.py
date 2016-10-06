#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
*********************************************
*
* SpectraLearnPredict
* Perform Machine Mearning on Raman data.
* version: 20161006a
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
import sys, os.path

#**********************************************
''' Spectra normalization conditions '''
#**********************************************
Ynorm = True
YnormTo = 1
YnormX = 1600
YnormXdelta = 30
# Normalize from the full spectra (False: recommended)
fullYnorm = False

#**********************************************
''' Energy normalization range '''
''' 150-250: D5; 450-550: G '''
#**********************************************
enRestrictRegion = False
enLim1 = 450    # for now use indexes rather than actual Energy
enLim2 = 550    # for now use indexes rather than actual Energy

#**********************************************
''' Support Vector Classification'''
#**********************************************
runSVM = True

''' Input/Output files '''
trainedData = "trained.pkl"
alwaysRetrain = False

''' Training algorithm
Use either 'linear' or 'rbf'
('rbf' for large number of features) '''

Cfactor = 10
kernel = 'rbf'
showClasses = False

showProbPlot = False
showTrainingDataPlot = False

#**********************************************
''' Neural Networks'''
#**********************************************
runNN = True

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
''' Main '''
#**********************************************
def main():
    try:
        LearnPredict(sys.argv[1], sys.argv[2])
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
    ''' Normalize if flag is set '''
    #**********************************************
    if Ynorm == True:
        print(' Normalizing spectral intensity to: ' + str(YnormTo) + '; En = [' + str(YnormX-YnormXdelta) + ', ' + str(YnormX+YnormXdelta) + ']\n')
        for i in range(0,A.shape[0]):
            Amax[i] = A[i,A[i][YnormXind].tolist().index(max(A[i][YnormXind].tolist()))+YnormXind[0]]
            A[i,:] = np.multiply(A[i,:], YnormTo/Amax[i])

    #**********************************************
    ''' Energy normalization range '''
    #**********************************************
    if enRestrictRegion == True:
        A = A[:,range(enLim1, enLim2)]
        En = En[range(enLim1, enLim2)]
        print( ' Restricting energy range between: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    else:
        print( ' Using full energy range: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')

    #**********************************************
    ''' Open prediction file '''
    #**********************************************
    try:
        with open(sampleFile, 'r') as f:
            print(' Opening sample data for prediction...')
            R = np.loadtxt(f, unpack =True, usecols=range(1,2))
    except:
        print('\033[1m' + '\n Sample data file not found \n ' + '\033[0m')
        return

    R = R.reshape(1,-1)
    if(R.shape[1] != AmaxIndex):
        print('\033[1m' + '\n WARNING: Prediction aborted. Different number of datapoints\n for the energy axis for training (' + str(AmaxIndex) + ') and sample (' + str(R.shape[1]) + ') data.\n Reformat sample data with ' + str(A.shape[1]) + ' X-datapoints.\n' + '\033[0m')
        return

    if Ynorm == True:
        Rmax = R[0,R[0][YnormXind].tolist().index(max(R[0][YnormXind].tolist()))+YnormXind[0]]
        R[0,:] = np.multiply(R[0,:], YnormTo/Rmax)

    if enRestrictRegion == True:
        R = R[:,range(enLim1, enLim2)]

    #***********************************
    ''' Run Support Vector Machines '''
    #***********************************
    if runSVM == True:
        runSVMmain(A, Cl, R)

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
        runTensorFlow(A,Cl,clf.classes_,R)


#********************
''' Run SVM '''
#********************
def runSVMmain(A, Cl, R):
    from sklearn import svm
    from sklearn.externals import joblib
    print('\n Running Support Vector Machine...')
    try:
        if alwaysRetrain == False:
            with open(trainedData):
                print(" Opening training data...")
                clf = joblib.load(trainedData)
        else:
            raise ValueError('Force retraining.')
    except:
        #**********************************************
        ''' Retrain data if not available'''
        #**********************************************
        print(' Retraining data...')
        clf = svm.SVC(kernel = kernel, C = Cfactor, decision_function_shape = 'ovr', probability=True)
        clf.fit(A,Cl)
        Z= clf.decision_function(A)
        print(' Number of classes = ' + str(Z.shape[1]))
        joblib.dump(clf, trainedData)
        if showClasses == True:
            print(' List of classes: ' + str(clf.classes_))

    print('\033[1m' + '\n Predicted value = ' + str(clf.predict(R)[0]) +'\n' + '\033[0m' )
    prob = clf.predict_proba(R)[0].tolist()

    #print(' Probabilities of this sample within each class: \n')
    #for i in range(0,clf.classes_.shape[0]):
    #   print(' ' + str(clf.classes_[i]) + ': ' + str(round(100*prob[i],2)) + '%')

    #********************
    ''' Plot results '''
    #********************
    if showProbPlot == True:
        import matplotlib.pyplot as plt
        print('\n Stand by: Plotting probabilities for each class... \n')
        plt.title('Probability density per class: ' + str(sampleFile))
        for i in range(0, clf.classes_.shape[0]):
            plt.scatter(clf.classes_[i], round(100*prob[i],2), label='probability', c = 'red')
        plt.grid(True)
        plt.xlabel('Class')
        plt.ylabel('Probability [%]')
        plt.show()

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

#*************************
''' Run Neural Network '''
#*************************
def runNNmain(A, Cl, R):
    from sklearn.neural_network import MLPClassifier
    print('\n Running Neural Network: multi-layer perceptron (MLP)...')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(A, Cl)
    print('\033[1m' + '\n Predicted value (Neural Networks) = ' + str(clf.predict(R)[0]) +'\n' + '\033[0m' )

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


#********************
''' Tensorflow '''
''' https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html'''
#********************
def runTensorFlow(A, Cl, Cl_label, R):
    import tensorflow as tf
    formatClassfile = "tfFormatClass.txt"
    
    try:
        with open(formatClassfile) as f:
            print(' Opening format class data...\n')
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
    print('\033[1m' + ' Prediction (TF): ' + str(Cl_label[res2][0]) + ' (' + str('{:.1f}'.format(res1[0][res2][0]*100)) + '%)\n' + '\033[0m' )

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:')
    print('  python SpectraLearnPredictSVM.py <mapfile> <spectrafile> \n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
