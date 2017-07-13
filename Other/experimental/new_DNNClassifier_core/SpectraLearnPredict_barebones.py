#!/usr/bin/env python3

import numpy as np
import sys, os.path
#**********************************************
''' Main '''
#**********************************************
def main():
    print(" Using training file: ", sys.argv[1],"\n")
        
    En, Cl, A = readLearnFile(sys.argv[1])
    En_test, Cl_test, A_test = readLearnFile(sys.argv[2])
    learnFileRoot = os.path.splitext(sys.argv[1])[0]
    clf_dnntf, le_dnntf  = trainDNNTF(A, Cl, A_test, Cl_test, learnFileRoot)

#**********************************************
''' DNNClassifier '''
#**********************************************
def trainDNNTF(A, Cl, A_test, Cl_test, Root):
    import tensorflow as tf
    import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing
    
    model_directory = Root + "/DNN-TF_"
    print("\n  Training model saved in: ", model_directory, "\n")
    
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
    
    feature_columns = skflow.infer_real_valued_columns_from_input(totA.astype(np.float32))
    
    ''' tf.estimator version '''
    #clf = tf.estimator.DNNClassifier(feature_columns=[totA], hidden_units=[20],
    #                           optimizer="Adagrad", n_classes=numTotClasses,
    #                           activation_fn="tanh", model_dir=model_directory)
    
    ''' tf.contrib.learn version '''
    clf = skflow.DNNClassifier(feature_columns=feature_columns, hidden_units=[20],
                                    optimizer="Adagrad", n_classes=numTotClasses,
                                    activation_fn="tanh", model_dir=model_directory)

    #**********************************************
    ''' Train '''
    #**********************************************
    
    ''' tf.estimator version '''
    #clf.train(input_fn=lambda: input_fn(A, Cl2), steps=2000)
    
    ''' tf.contrib.learn version '''
    clf.fit(input_fn=lambda: input_fn(A, Cl2), steps=100)
    
    
    accuracy_score = clf.evaluate(input_fn=lambda: input_fn(A_test, Cl2_test), steps=1)
    print('\n  ================================')
    print('  \033[1mDNN-TF\033[0m - Accuracy')
    print('  ================================')
    print("\n  Accuracy: {:.2f}%".format(100*accuracy_score["accuracy"]))
    print("  Loss: {:.2f}".format(accuracy_score["loss"]))
    print("  Global step: {:.2f}\n".format(accuracy_score["global_step"]))
    print('  ================================\n')

    return clf, le

#**********************************************
''' Format input data for Estimator '''
#**********************************************
def input_fn(A, Cl2):
    import tensorflow as tf
    x = tf.constant(A.astype(np.float32))
    y = tf.constant(Cl2)
    return x,y

#**********************************************
''' Read learn File '''
#**********************************************
def readLearnFile(learnFile):
    try:
        with open(learnFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Learning file not found \n' + '\033[0m')
        return
    
    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(M,np.s_[0:1],0)
    Cl = ['{:.2f}'.format(x) for x in M[:,0]]
    A = np.delete(M,np.s_[0:1],1)
    return En, Cl, A

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
