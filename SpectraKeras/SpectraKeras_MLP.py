#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
* SpectraKeras_MLP Classifier and Regressor
* 20181120a
* Uses: Keras, TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, getopt, time, configparser, pickle, h5py, csv
from libSpectraKeras import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def SpectraKeras_MLP():
    main()

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self):
        confFileName = "SpectraKeras_MLP.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
            
    def SKDef(self):
        self.conf['Parameters'] = {
            'regressor' : False,
            'normalize' : False,
            'l_rate' : 0.001,
            'l_rdecay' : 1e-4,
            'HL' : [20,30,40,50,60,70],
            'drop' : 0,
            'l2' : 1e-4,
            'epochs' : 100,
            'cv_split' : 0.01,
            'fullSizeBatch' : True,
            'batch_size' : 64,
            'numLabels' : 1,
            'plotWeightsFlag' : False,
            }
    def sysDef(self):
        self.conf['System'] = {
            'useTFKeras' : False,
            }

    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.SKDef = self.conf['Parameters']
            self.sysDef = self.conf['System']
        
            self.regressor = self.conf.getboolean('Parameters','regressor')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.HL = eval(self.SKDef['HL'])
            self.drop = self.conf.getfloat('Parameters','drop')
            self.l2 = self.conf.getfloat('Parameters','l2')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.cv_split = self.conf.getfloat('Parameters','cv_split')
            self.fullSizeBatch = self.conf.getboolean('Parameters','fullSizeBatch')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.numLabels = self.conf.getint('Parameters','numLabels')
            self.plotWeightsFlag = self.conf.getboolean('Parameters','plotWeightsFlag')
            self.useTFKeras = self.conf.getboolean('System','useTFKeras')
        except:
            print(" Error in reading configuration file. Please check it\n")

    # Create configuration file
    def createConfig(self):
        try:
            self.SKDef()
            self.sysDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except:
            print("Error in creating configuration file")

#************************************
# Main
#************************************
def main():
    start_time = time.clock()
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "tpbh:", ["train", "predict", "batch", "help"])
    except:
        usage()
        sys.exit(2)

    if opts == []:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-t" , "--train"):
            #try:
            if len(sys.argv)<4:
                train(sys.argv[2], None)
            else:
                train(sys.argv[2], sys.argv[3])
            #except:
            #   usage()
            #   sys.exit(2)

        if o in ("-p" , "--predict"):
            try:
                predict(sys.argv[2])
            except:
                usage()
                sys.exit(2)
                
        if o in ("-b" , "--batch"):
            try:
                if len(sys.argv)<4:
                    batchPredict(sys.argv[2], None)
                else:
                    batchPredict(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)

    total_time = time.clock() - start_time
    print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

#************************************
# Training
#************************************
def train(learnFile, testFile):
    import tensorflow as tf
    dP = Conf()
    
    # Use this to restrict GPU memory allocation in TF
    opts = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    conf = tf.ConfigProto(gpu_options=opts)
    #conf.gpu_options.allow_growth = True
    
    if dP.useTFKeras:
        print("Using tf.keras API")
        import tensorflow.keras as keras  #tf.keras
        tf.Session(config=conf)
    else:
        print("Using pure keras API")
        import keras   # pure keras
        from keras.backend.tensorflow_backend import set_session
        set_session(tf.Session(config=conf))

    tb_directory = "keras_MLP"
    model_directory = "."
    learnFileRoot = os.path.splitext(learnFile)[0]

    if dP.regressor:
        print("Using SpectraKeras_MLP as Regressor")
        model_name = model_directory+"/keras_model_regressor.hd5"
    else:
        print("Using SpectraKeras_MLP as Classifier")
        model_name = model_directory+"/keras_model.hd5"
        model_le = model_directory+"/keras_le.pkl"

    #from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

    En, A, Cl = readLearnFile(learnFile)
    if testFile != None:
        En_test, A_test, Cl_test = readLearnFile(testFile)
        totA = np.vstack((A, A_test))
        totCl = np.append(Cl, Cl_test)
    else:
        totA = A
        totCl = Cl

    print("  Total number of points per data:",En.size)
    print("  Number of learning labels: {0:d}\n".format(int(dP.numLabels)))
    
    if dP.regressor:
        Cl2 = np.copy(Cl)
        if testFile != None:
            Cl2_test = np.copy(Cl_test)
    else:
    
        #************************************
        # Label Encoding
        #************************************
        '''
        # sklearn preprocessing is only for single labels
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        totCl2 = le.fit_transform(totCl)
        Cl2 = le.transform(Cl)
        if testFile != None:
            Cl2_test = le.transform(Cl_test)
        '''
        le = MultiClassReductor()
        le.fit(np.unique(totCl, axis=0))
        Cl2 = le.transform(Cl)
    
        print("  Number unique classes (training): ", np.unique(Cl).size)
    
        if testFile != None:
            Cl2_test = le.transform(Cl_test)
            print("  Number unique classes (validation):", np.unique(Cl_test).size)
            print("  Number unique classes (total): ", np.unique(totCl).size)
            
        print("\n  Label Encoder saved in:", model_le,"\n")
        with open(model_le, 'ab') as f:
            f.write(pickle.dumps(le))

        #totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
        Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(totCl).size+1)
        if testFile != None:
            Cl2_test = keras.utils.to_categorical(Cl2_test, num_classes=np.unique(totCl).size+1)

    #************************************
    # Training
    #************************************

    if dP.fullSizeBatch == True:
        dP.batch_size = A.shape[0]

    #************************************
    ### Define optimizer
    #************************************
    #optim = opt.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    optim = keras.optimizers.Adam(lr=dP.l_rate, beta_1=0.9,
                    beta_2=0.999, epsilon=1e-08,
                    decay=dP.l_rdecay,
                    amsgrad=False)
    #************************************
    ### Build model
    #************************************
    model = keras.models.Sequential()
    for i in range(len(dP.HL)):
        model.add(keras.layers.Dense(dP.HL[i],
            activation = 'relu',
            input_dim=A.shape[1],
            kernel_regularizer=keras.regularizers.l2(dP.l2)))
        model.add(keras.layers.Dropout(dP.drop))

    if dP.regressor:
        model.add(keras.layers.Dense(1))
        model.compile(loss='mse',
        optimizer=optim,
        metrics=['mae'])
    else:
        model.add(keras.layers.Dense(np.unique(totCl).size+1, activation = 'softmax'))
        model.compile(loss='categorical_crossentropy',
            optimizer=optim,
            metrics=['accuracy'])

    tbLog = keras.callbacks.TensorBoard(log_dir=tb_directory, histogram_freq=120,
            batch_size=dP.batch_size,
            write_graph=True, write_grads=True, write_images=True)
    tbLogs = [tbLog]
    if testFile != None:
        log = model.fit(A, Cl2,
            epochs=dP.epochs,
            batch_size=dP.batch_size,
            callbacks = tbLogs,
            verbose=2,
            validation_data=(A_test, Cl2_test))
    else:
        log = model.fit(A, Cl2,
            epochs=dP.epochs,
            batch_size=dP.batch_size,
            callbacks = tbLogs,
            verbose=2,
	        validation_split=dP.cv_split)

    model.save(model_name)
    keras.utils.plot_model(model, to_file=model_directory+'/keras_MLP_model.png', show_shapes=True)

    print('\n  =============================================')
    print('  \033[1mKeras MLP\033[0m - Model Configuration')
    print('  =============================================')
    #for conf in model.get_config():
    #    print(conf,"\n")
    print("  Training set file:",learnFile)
    print("  Data size:", A.shape,"\n")
    print("  Number of learning labels:",dP.numLabels)
    print("  Total number of points per data:",En.size)

    loss = np.asarray(log.history['loss'])
    val_loss = np.asarray(log.history['val_loss'])

    if dP.regressor:
        val_mae = np.asarray(log.history['val_mean_absolute_error'])
        predictions = model.predict(A_test)
        printParam()
        print('\n  ==========================================================')
        print('  \033[1mKeras MLP - Regressor\033[0m - Training Summary')
        print('  ==========================================================')
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(loss), np.amin(loss), loss[-1]))
        print('\n\n  ==========================================================')
        print('  \033[1mKeras MLP - Regressor \033[0m - Validation Summary')
        print('  ========================================================')
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(val_loss), np.amin(val_loss), val_loss[-1]))
        print("  \033[1mMean Abs Err\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}\n".format(np.average(val_mae), np.amin(val_mae), val_mae[-1]))
        
        if testFile != None:
            print('  ========================================================')
            print("  Real value | Predicted value | val_loss | val_mean_abs_err")
            print("  -----------------------------------------------------------")
            for i in range(0,len(predictions)):
                score = model.evaluate(np.array([A_test[i]]), np.array([Cl_test[i]]), batch_size=dP.batch_size, verbose = 0)
                print("  {0:.2f}\t\t| {1:.2f}\t\t| {2:.4f}\t| {3:.4f} ".format(Cl2_test[i],
                    predictions[i][0], score[0], score[1]))
            print('\n  ==========================================================\n')
    else:
        accuracy = np.asarray(log.history['acc'])
        val_acc = np.asarray(log.history['val_acc'])
        print("  Number unique classes (training): ", np.unique(Cl).size)
        if testFile != None:
            Cl2_test = le.transform(Cl_test)
            print("  Number unique classes (validation):", np.unique(Cl_test).size)
            print("  Number unique classes (total): ", np.unique(totCl).size)
        printParam()
        print('\n  ========================================================')
        print('  \033[1mKeras MLP - Classiefier \033[0m - Training Summary')
        print('  ========================================================')
        print("\n  \033[1mAccuracy\033[0m - Average: {0:.2f}%; Max: {1:.2f}%; Last: {2:.2f}%".format(100*np.average(accuracy),
            100*np.amax(accuracy), 100*accuracy[-1]))
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(loss), np.amin(loss), loss[-1]))
        print('\n\n  ========================================================')
        print('  \033[1mKeras MLP - Classifier \033[0m - Validation Summary')
        print('  ========================================================')
        print("\n  \033[1mAccuracy\033[0m - Average: {0:.2f}%; Max: {1:.2f}%; Last: {2:.2f}%".format(100*np.average(val_acc),
        100*np.amax(val_acc), 100*val_acc[-1]))
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}\n".format(np.average(val_loss), np.amin(val_loss), val_loss[-1]))
        if testFile != None:
            print('  ========================================================')
            print("  Real class\t| Predicted class\t| Probability")
            print("  ---------------------------------------------------")
            predictions = model.predict(A_test)
            for i in range(predictions.shape[0]):
                predClass = np.argmax(predictions[i])
                predProb = round(100*predictions[i][predClass],2)
                predValue = le.inverse_transform([predClass])[0]
                realValue = Cl_test[i]
                print("  {0:.2f}\t\t| {1:.2f}\t\t\t| {2:.2f}".format(realValue, predValue, predProb))
            #print("\n  Validation - Loss: {0:.2f}; accuracy: {1:.2f}%".format(score[0], 100*score[1]))
            print('\n  ========================================================\n')

    if dP.plotWeightsFlag == True:
        plotWeights(En, A, model)

#************************************
# Prediction
#************************************
def predict(testFile):
    dP = Conf()
    if dP.useTFKeras:
        import tensorflow.keras as keras  #tf.keras
    else:
        import keras   # pure Keras
    
    try:
        with open(testFile, 'r') as f:
            print('\n Opening sample data for prediction...')
            Rtot = np.loadtxt(f, unpack =True)
    except:
        print('\033[1m' + '\n Sample data file not found \n ' + '\033[0m')
        return

    R=np.array([Rtot[1,:]])
    Rx=Rtot[0,:]

    if dP.normalize:
        norm = Normalizer()
        R = norm.transform_single(R)
    if dP.regressor:
        model = keras.models.load_model("keras_model_regressor.hd5")
        predictions = model.predict(R).flatten()[0]
        print('  ========================================================')
        print('  \033[1mKeras MLP - Regressor\033[0m - Prediction')
        print('  ========================================================')
        predValue = predictions
        print('\033[1m\n  Predicted value (normalized) = {0:.2f}\033[0m\n'.format(predValue))
        print('  ========================================================\n')
        
    else:
        le = pickle.loads(open("keras_le.pkl", "rb").read())
        model = keras.models.load_model("keras_model.hd5")
        predictions = model.predict(R, verbose=1)
        pred_class = np.argmax(predictions)
        predProb = round(100*predictions[0][pred_class],2)
        rosterPred = np.where(predictions[0]>0.1)[0]

        print('  ========================================================')
        print('  \033[1mKeras MLP - Classifier\033[0m - Prediction')
        print('  ========================================================')

        if dP.numLabels == 1:

            if pred_class.size >0:
                predValue = le.inverse_transform([pred_class])[0]
                print('\033[1m\n  Predicted value (normalized) = {0:.2f} (probability = {1:.2f}%)\033[0m\n'.format(predValue, predProb))
            else:
                predValue = 0
                print('\033[1m\n  No predicted value (probability = {0:.2f}%)\033[0m\n'.format(predProb))
            print('  ========================================================\n')

        else:
            print('\n ==========================================')
            print('\033[1m' + ' Predicted value \033[0m(probability = ' + str(predProb) + '%)')
            print(' ==========================================\n')
            print("  1:", str(predValue[0]),"%")
            print("  2:",str(predValue[1]),"%")
            print("  3:",str((predValue[1]/0.5)*(100-99.2-.3)),"%\n")
            print(' ==========================================\n')

#************************************
# Batch Prediction
#************************************
def batchPredict(testFile, normFile):
    dP = Conf()
    if dP.useTFKeras:
        import tensorflow.keras as keras  #tf.keras
    else:
        import keras   # pure Keras

    En_test, A_test, Cl_test = readLearnFile(testFile)

    if normFile != None:
        try:
            norm = pickle.loads(open(normFile, "rb").read())
            print("\n  Opening pkl file with normalization data:",normFile,"\n")
        except:
            print("\033[1m  pkl file not found\033[0m \n")
            return

    if dP.regressor:
        summaryFileName = os.path.splitext(testFile)[0]+"_regressor-summary.csv"
        summaryFile = np.array([['SpectraKeras_MLP','Regressor','',''],['Real Value','Prediction','val_loss','val_abs_mean_error']])
        model = keras.models.load_model("keras_model_regressor.hd5")
        predictions = model.predict(A_test)
        score = model.evaluate(A_test, Cl_test, batch_size=dP.batch_size, verbose = 0)
        print('  ==========================================================')
        print('  \033[1mKeras MLP - Regressor\033[0m - Batch Prediction')
        print('  ==========================================================')
        print("  \033[1mOverall val_loss:\033[0m {0:.4f}; \033[1moverall val_abs_mean_loss:\033[0m {1:.4f}\n".format(score[0], score[1]))
        print('  ==========================================================')
        print("  Real value | Predicted value | val_loss | val_mean_abs_err")
        print("  -----------------------------------------------------------")
        for i in range(0,len(predictions)):
            score = model.evaluate(np.array([A_test[i]]), np.array([Cl_test[i]]), batch_size=dP.batch_size, verbose = 0)
            if normFile != None:
                predValue = norm.transform_inverse_single(predictions[i][0])
                realValue = norm.transform_inverse_single(Cl_test[i])
            else:
                realValue = Cl_test[i]
                predValue = predictions[i][0]
            
            print("  {0:.2f}\t\t| {1:.2f}\t\t| {2:.4f}\t| {3:.4f} ".format(realValue, predValue, score[0], score[1]))
            summaryFile = np.vstack((summaryFile,[realValue,predValue,score[0], score[1]]))
        print('  ==========================================================\n')
    else:
        summaryFileName = os.path.splitext(testFile)[0]+"_classifier-summary.csv"
        summaryFile = np.array([['SpectraKeras_MLP','Classifier',''],['Real Class','Predicted Class', 'Probability']])
        le = pickle.loads(open("keras_le.pkl", "rb").read())
        model = keras.models.load_model("keras_model.hd5")
        predictions = model.predict(A_test)
        print('  ========================================================')
        print('  \033[1mKeras MLP - Classifier\033[0m - Batch Prediction')
        print('  ========================================================')
        print("  Real class\t| Predicted class\t| Probability")
        print("  ---------------------------------------------------")
        for i in range(predictions.shape[0]):
            predClass = np.argmax(predictions[i])
            predProb = round(100*predictions[i][predClass],2)
            if normFile != None:
                predValue = norm.transform_inverse_single(le.inverse_transform([predClass])[0])
                realValue = norm.transform_inverse_single(Cl_test[i])
            else:
                predValue = le.inverse_transform([predClass])[0]
                realValue = Cl_test[i]
            print("  {0:.2f}\t\t| {1:.2f}\t\t\t| {2:.2f}".format(realValue, predValue, predProb))
            summaryFile = np.vstack((summaryFile,[realValue,predValue,predProb]))
        print('  ========================================================\n')

    df = pd.DataFrame(summaryFile)
    df.to_csv(summaryFileName, index=False, header=False)
    print(" Prediction summary saved in:",summaryFileName,"\n")

#************************************
# Open Learning Data
#************************************
def readLearnFile(learnFile):
    print("\n  Opening learning file: ",learnFile)
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
        print("\033[1m Learning file not found\033[0m")
        return

    dP = Conf()
    En = M[0,dP.numLabels:]
    A = M[1:,dP.numLabels:]
    
    if dP.normalize:
        norm = Normalizer()
        A = norm.transform_matrix(A)

    if dP.numLabels == 1:
        Cl = M[1:,0]
    else:
        Cl = M[1:,[0,dP.numLabels-1]]

    return En, A, Cl

#************************************
# Print NN Info
#************************************
def printParam():
    dP = Conf()
    print('\n  ================================================')
    print('  \033[1mKeras MLP\033[0m - Parameters')
    print('  ================================================')
    print('  Optimizer:','Adam',
                '\n  Hidden layers:', dP.HL,
                '\n  Activation function:','relu',
                '\n  L2:',dP.l2,
                '\n  Dropout:', dP.drop,
                '\n  Learning rate:', dP.l_rate,
                '\n  Learning decay rate:', dP.l_rdecay)
    if dP.fullSizeBatch == True:
        print('  Batch size: full')
    else:
        print('  Batch size:', dP.batch_size)
    print('  Number of labels:', dP.numLabels)
    #print('  ================================================\n')

#************************************
# Open Learning Data
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
# Lists the program usage
#************************************
def usage():
    print('\n Usage:\n')
    print(' Train (Random cross validation):')
    print('  python3 SpectraKeras_MLP.py -t <learningFile>\n')
    print(' Train (with external validation):')
    print('  python3 SpectraKeras_MLP.py -t <learningFile> <validationFile>\n')
    print(' Predict (no label normalization used):')
    print('  python3 SpectraKeras_MLP.py -p <testFile>\n')
    print(' Predict (labels normalized with pkl file):')
    print('  python3 SpectraKeras_MLP.py -p <testFile> <pkl normalization file>\n')
    print(' Batch predict (no label normalization used):')
    print('  python3 SpectraKeras_MLP.py -b <validationFile>\n')
    print(' Batch predict (labels normalized with pkl file):')
    print('  python3 SpectraKeras_MLP.py -b <validationFile> <pkl normalization file>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
