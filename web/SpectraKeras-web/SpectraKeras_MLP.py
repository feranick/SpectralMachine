#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************
* SpectraKeras_MLP Classifier and Regressor
* v2025.05.21.1
* Uses: TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
**********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, getopt, time, configparser, ast
import platform, pickle, h5py, csv, glob
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
        self.configFile = os.path.join(os.getcwd(),confFileName)
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"
        if self.regressor:
            self.modelName = "model_regressor_MLP.h5"
            self.summaryFileName = "summary_regressor_MLP.csv"
            self.model_png = self.model_directory+"model_regressor_MLP.png"
        else:
            self.predProbThreshold = 90.00
            self.modelName = "model_classifier_MLP.h5"
            self.summaryFileName = "summary_classifier_MLP.csv"
            self.summaryAccFileName = "summary_classifier_MLP_accuracy.csv"
            self.model_png = self.model_directory+"model_classifier_MLP.png"

        self.tb_directory = "model_MLP"
        self.model_name = self.model_directory+self.modelName
        
        if self.kerasVersion == 3:
            self.model_name = os.path.splitext(self.model_name)[0]+".keras"
        
        self.model_le = self.model_directory+"model_le.pkl"
        self.spectral_range = "model_spectral_range.pkl"
        self.table_names = self.model_directory+"AAA_table_names.h5"

        if platform.system() == 'Linux':
            self.edgeTPUSharedLib = "libedgetpu.so.1"
        if platform.system() == 'Darwin':
            self.edgeTPUSharedLib = "libedgetpu.1.dylib"
        if platform.system() == 'Windows':
            self.edgeTPUSharedLib = "edgetpu.dll"

    def SKDef(self):
        self.conf['Parameters'] = {
            'regressor' : False,
            'normalize' : True,
            'l_rate' : 0.001,
            'l_rdecay' : 0.96,
            'HL' : [20,30,40,50,60,70],
            'drop' : 0,
            'l2' : 1e-4,
            'epochs' : 100,
            'cv_split' : 0.01,
            'fullSizeBatch' : False,
            'batch_size' : 64,
            'numLabels' : 1,
            'plotWeightsFlag' : False,
            'showValidPred' : False,
            'stopAtBest' : False,
            'saveBestModel' : False,
            'metricBestModelR' : 'val_mae',
            'metricBestModelC' : 'val_accuracy',
            }

    def sysDef(self):
        self.conf['System'] = {
            'kerasVersion' : 2,
            'makeQuantizedTFlite' : True,
            'useTFlitePred' : False,
            'TFliteRuntime' : False,
            'runCoralEdge' : False,
            }

    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.SKPar = self.conf['Parameters']
            self.sysPar = self.conf['System']

            self.regressor = self.conf.getboolean('Parameters','regressor')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.HL = ast.literal_eval(self.SKPar['HL'])
            self.drop = self.conf.getfloat('Parameters','drop')
            self.l2 = self.conf.getfloat('Parameters','l2')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.cv_split = self.conf.getfloat('Parameters','cv_split')
            self.fullSizeBatch = self.conf.getboolean('Parameters','fullSizeBatch')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.numLabels = self.conf.getint('Parameters','numLabels')
            self.plotWeightsFlag = self.conf.getboolean('Parameters','plotWeightsFlag')
            self.showValidPred = self.conf.getboolean('Parameters','showValidPred')
            self.stopAtBest = self.conf.getboolean('Parameters','stopAtBest')
            self.saveBestModel = self.conf.getboolean('Parameters','saveBestModel')
            self.metricBestModelR = self.conf.get('Parameters','metricBestModelR')
            self.metricBestModelC = self.conf.get('Parameters','metricBestModelC')
            
            self.kerasVersion = self.conf.getint('System','kerasVersion')
            self.makeQuantizedTFlite = self.conf.getboolean('System','makeQuantizedTFlite')
            self.useTFlitePred = self.conf.getboolean('System','useTFlitePred')
            self.TFliteRuntime = self.conf.getboolean('System','TFliteRuntime')
            self.runCoralEdge = self.conf.getboolean('System','runCoralEdge')
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
    start_time = time.perf_counter()
    dP = Conf()

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "tpblah:", ["train", "predict", "lite", "batch", "accuracy", "help"])
    except:
        usage()
        sys.exit(2)

    if opts == []:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-t" , "--train"):
            try:
                if len(sys.argv)<4:
                    train(sys.argv[2], None)
                else:
                    train(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)

        if o in ("-p" , "--predict"):
            try:
                predict(sys.argv[2])
            except:
               usage()
               sys.exit(2)

        if o in ("-b" , "--batch"):
            try:
                batchPredict(sys.argv[2])
            except:
                usage()
                sys.exit(2)

        if o in ("-l" , "--lite"):
            try:
                convertTflite(sys.argv[2])
            except:
                usage()
                sys.exit(2)

        if o in ("-a" , "--accuracy"):
            try:
                accDeterm(sys.argv[2])
            except:
                usage()
                sys.exit(2)

    total_time = time.perf_counter() - start_time
    print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

#************************************
# Training
#************************************
def train(learnFile, testFile):
    dP = Conf()
    import tensorflow as tf
    if checkTFVersion("2.16.0"):
        import tensorflow.keras as keras
    else:
        if dP.kerasVersion == 2:
            import tf_keras as keras
        else:
            import keras

    opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1)     # Tensorflow 2.0
    conf = tf.compat.v1.ConfigProto(gpu_options=opts)  # Tensorflow 2.0

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    #   if dP.setMaxMem:
    #       tf.config.set_virtual_device_configuration(
    #         gpus[0],
    #         [tf.config.VirtualDeviceConfiguration(memory_limit=dP.maxMem)])

    learnFileRoot = os.path.splitext(learnFile)[0]

    En, A, Cl = readLearnFile(learnFile, dP)
    if testFile is not None:
        En_test, A_test, Cl_test = readLearnFile(testFile, dP)
        totA = np.vstack((A, A_test))
        totCl = np.append(Cl, Cl_test)
    else:
        totA = A
        totCl = Cl

    with open(dP.spectral_range, 'wb') as f:
        pickle.dump(En, f)

    print("  Total number of points per data:",En.size)
    print("  Number of learning labels: {0:d}\n".format(int(dP.numLabels)))

    if dP.regressor:
        Cl2 = np.copy(Cl)
        if testFile is not None:
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
        if testFile is not None:
            Cl2_test = le.transform(Cl_test)
        '''
        le = MultiClassReductor()
        le.fit(np.unique(totCl, axis=0))
        Cl2 = le.transform(Cl)

        print("  Number unique classes (training): ", np.unique(Cl).size)

        if testFile is not None:
            Cl2_test = le.transform(Cl_test)
            print("  Number unique classes (validation):", np.unique(Cl_test).size)
            print("  Number unique classes (total): ", np.unique(totCl).size)

        print("\n  Label Encoder saved in:", dP.model_le,"\n")
        with open(dP.model_le, 'wb') as f:
            pickle.dump(le, f)

        #totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
        Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(totCl).size+1)
        if testFile is not None:
            Cl2_test = keras.utils.to_categorical(Cl2_test, num_classes=np.unique(totCl).size+1)

    #************************************
    # Training
    #************************************

    if dP.fullSizeBatch:
        dP.batch_size = A.shape[0]

    #************************************
    ### Build model
    #************************************
    def get_model():
        
        #************************************
        ### Define optimizer
        #************************************
        #optim = opt.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=dP.l_rate,
            decay_steps=dP.epochs,
            decay_rate=dP.l_rdecay)
        optim = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9,
                beta_2=0.999, epsilon=1e-08,
                amsgrad=False)
                    
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(A.shape[1],)))
        for i in range(len(dP.HL)):
            model.add(keras.layers.Dense(dP.HL[i],
                activation = 'relu',
                #input_dim=A.shape[1],
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
        return model
        
    model = get_model()

    tbLog = keras.callbacks.TensorBoard(log_dir=dP.tb_directory, histogram_freq=120,
            batch_size=dP.batch_size,
            write_graph=True, write_grads=True, write_images=False)
    
    tbLogs = [tbLog]
    if dP.stopAtBest == True:
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)
        tbLogs.append(es)
    if dP.saveBestModel == True:
        if dP.regressor:
            mc = keras.callbacks.ModelCheckpoint(dP.model_name, monitor=dP.metricBestModelR, mode='min', verbose=1, save_best_only=True)
        else:
            mc = keras.callbacks.ModelCheckpoint(dP.model_name, monitor=dP.metricBestModelC, mode='max', verbose=1, save_best_only=True)
        tbLogs.append(mc)
        
    #tbLogs = [tbLog, es, mc]

    print('  =============================================')
    print('  \033[1m MLP\033[0m - Model Architecture')
    print('  =============================================\n')
    model.summary()

    if testFile is not None:
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
         
    if dP.saveBestModel == False:
        model.save(dP.model_name)
    else:
        model = loadModel(dP)
    
    keras.utils.plot_model(model, to_file=dP.model_png, show_shapes=True)

    if dP.makeQuantizedTFlite:
        makeQuantizedTFmodel(A, dP)

    print('\n  =============================================')
    print('  \033[1m MLP\033[0m - Model Architecture')
    print('  =============================================\n')
    model.summary()

    print('\n  ========================================================')
    print('  \033[1m MLP\033[0m - Training/Validation set Configuration')
    print('  ========================================================')

    print("  Training set file:",learnFile)
    if testFile is not None:
        print("  Validation set file:",testFile)
    print("  Data size:", A.shape,"\n")
    print("  Number of learning labels:",dP.numLabels)
    print("  Total number of points per data:",En.size)

    loss = np.asarray(log.history['loss'])
    val_loss = np.asarray(log.history['val_loss'])

    if dP.regressor:
        mae = np.asarray(log.history['mae'])
        val_mae = np.asarray(log.history['val_mae'])
        printParam()
        print('\n  ==========================================================')
        print('  \033[1m MLP - Regressor\033[0m - Training Summary')
        print('  ==========================================================')
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(loss), np.amin(loss), loss[-1]))
        print("  \033[1mMean Abs Err\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(mae), np.amin(mae), mae[-1]))
        print('\n\n  ==========================================================')
        print('  \033[1m MLP - Regressor \033[0m - Validation Summary')
        print('  ========================================================')
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(val_loss), np.amin(val_loss), val_loss[-1]))
        print("  \033[1mMean Abs Err\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(val_mae), np.amin(val_mae), val_mae[-1]))
        if dP.saveBestModel:
            if dP.metricBestModelR == 'mae':
                if testFile:
                    score = model.evaluate(A_test, Cl_test, batch_size=dP.batch_size, verbose = 0)
                    print("  \033[1mSaved model with validation MAE:\033[0m: {0:.4f}".format(score[1]))
                print("  \033[1mSaved model with min training MAE:\033[0m: {0:.4f}\n".format(np.amin(mae)))
            if dP.metricBestModelR == 'val_mae':
                print("  \033[1mSaved model with validation MAE:\033[0m: {0:.4f}\n".format(np.amin(val_mae)))
            else:
                pass
        print('  ========================================================\n')
        if testFile is not None and dP.showValidPred:
            predictions = model.predict(A_test)
            print("  Real value | Predicted value | val_loss | val_mean_abs_err")
            print("  -----------------------------------------------------------")
            for i in range(0,len(predictions)):
                score = model.evaluate(np.array([A_test[i]]), np.array([Cl_test[i]]), batch_size=dP.batch_size, verbose = 0)
                print("  {0:.2f}\t\t| {1:.2f}\t\t| {2:.4f}\t| {3:.4f} ".format(Cl2_test[i],
                    predictions[i][0], score[0], score[1]))
            print('\n  ==========================================================\n')
    else:
        accuracy = np.asarray(log.history['accuracy'])
        val_acc = np.asarray(log.history['val_accuracy'])
        print("  Number unique classes (training): ", np.unique(Cl).size)
        printParam()
        print('\n  ========================================================')
        print('  \033[1m MLP - Classifier \033[0m - Training Summary')
        print('  ========================================================')
        print("\n  \033[1mAccuracy\033[0m - Average: {0:.2f}%; Max: {1:.2f}%; Last: {2:.2f}%".format(100*np.average(accuracy),
            100*np.amax(accuracy), 100*accuracy[-1]))
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(loss), np.amin(loss), loss[-1]))
        print('\n\n  ========================================================')
        print('  \033[1m MLP - Classifier \033[0m - Validation Summary')
        print('  ========================================================')
        print("\n  \033[1mAccuracy\033[0m - Average: {0:.2f}%; Max: {1:.2f}%; Last: {2:.2f}%".format(100*np.average(val_acc),
        100*np.amax(val_acc), 100*val_acc[-1]))
        print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(val_loss), np.amin(val_loss), val_loss[-1]))
        if dP.saveBestModel:
            if dP.metricBestModelC == 'accuracy':
                print("  \033[1mSaved model with training accuracy:\033[0m: {0:.4f}".format(100*np.amax(accuracy)))
            if dP.metricBestModelC == 'val_acc':
                print("  \033[1mSaved model with validation accuracy:\033[0m: {0:.4f}\n".format(100*np.amax(val_acc)))
            else:
                pass
        print('  ========================================================\n')
        if testFile is not None and dP.showValidPred:
            print("  Real class\t| Predicted class\t| Probability")
            print("  ---------------------------------------------------")
            predictions = model.predict(A_test)
            for i in range(predictions.shape[0]):
                predClass = np.argmax(predictions[i])
                predProb = round(100*predictions[i][predClass],2)
                predValue = le.inverse_transform(predClass)[0]
                realValue = Cl_test[i]
                print("  {0:.2f}\t\t| {1:.2f}\t\t\t| {2:.2f}".format(realValue, predValue, predProb))
            #print("\n  Validation - Loss: {0:.2f}; accuracy: {1:.2f}%".format(score[0], 100*score[1]))
            print('\n  ========================================================\n')

    if dP.plotWeightsFlag:
        plotWeights(dP, En, A, model, "MLP")

    getTFVersion(dP)

#************************************
# Prediction
#************************************
def predict(testFile):
    dP = Conf()
    model = loadModel(dP)

    with open(dP.spectral_range, "rb") as f:
        EnN = pickle.load(f)

    R, good = readTestFile(testFile, EnN, dP)
    if not good:
        return

    if dP.regressor:
        #predictions = model.predict(R).flatten()[0]
        predictions, _ = getPredictions(R, model, dP)
        print('\n  ========================================================')
        print('  \033[1m MLP - Regressor\033[0m - Prediction')
        print('  ========================================================')
        predValue = predictions.flatten()[0]
        print('\033[1m\n  Predicted value (normalized) = {0:.2f}\033[0m\n'.format(predValue))
        print('  ========================================================\n')

    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        #predictions = model.predict(R, verbose=0)
        predictions, _ = getPredictions(R, model,dP)
        pred_class = np.argmax(predictions)
        if dP.useTFlitePred:
            predProb = round(100*predictions[0][pred_class]/255,2)
        else:
            predProb = round(100*predictions[0][pred_class],2)
        rosterPred = np.where(predictions[0]>0.1)[0]
        print('\n  ========================================================')
        print('  \033[1mK MLP - Classifier\033[0m - Prediction')
        print('  ========================================================')

        if dP.numLabels == 1:
            if pred_class.size >0:
                predValue = le.inverse_transform(pred_class)[0]
            else:
                predValue = 0
            print('  Prediction\t| Class \t| Probability [%]')
            print('  -------------------------------------------------------- ')
            for i in range(len(predictions[0])-1):
                if predictions[0][i]>0.01:
                    if dP.useTFlitePred:
                        print("  {0:s}\t| {1:d}\t\t| {2:.2f}".format(getMineral(dP.table_names, int(le.inverse_transform(i)[0])),
                            int(le.inverse_transform(i)[0]), 100*predictions[0][i]/255))
                    else:
                        print("  {0:s}\t| {1:d}\t\t| {2:.2f}".format(getMineral(dP.table_names, int(le.inverse_transform(i)[0])),
                            int(le.inverse_transform(i)[0]), 100*predictions[0][i]))
                    
            print('\033[1m\n  {0:s} \033[0m(Class: {1:d}, probability = {2:.2f}%)\n'.format(getMineral(dP.table_names, int(predValue)), int(predValue), predProb))
            print('  ========================================================\n')

        else:
            '''
            print('\n ==========================================')
            print('\033[1m' + ' Predicted value \033[0m(probability = ' + str(predProb) + '%)')
            print(' ==========================================\n')
            print("  1:", str(predValue[0]),"%")
            print("  2:",str(predValue[1]),"%")
            print("  3:",str((predValue[1]/0.5)*(100-99.2-.3)),"%\n")
            print(' ==========================================\n')
            '''

#************************************
# Batch Prediction
#************************************
def batchPredict(folder):
    dP = Conf()
    model = loadModel(dP)
    with open(dP.spectral_range, "rb") as f:
        EnN = pickle.load(f)

    predictions = np.zeros((0,0))
    fileName = []
    predictions_list = []
    for file in glob.glob(folder+'/*.txt'):
        R, good = readTestFile(file, EnN, dP)
        if good:
            R = formatForCNN(R)
            try:
                predictions_list.append(getPredictions(R, model, dP)[0].flatten())
            except:
                predictions_list = [np.array([getPredictions(R, model,dP)[0].flatten()])]
            fileName.append(file)
    predictions = np.array(predictions_list)

    if dP.regressor:
        summaryFile = np.array([['SpectraKeras_MLP','Regressor','',],['File name','Prediction','']])
        print('\n  ========================================================')
        print('  \033[1mKeras MLP - Regressor\033[0m - Prediction')
        print('  ========================================================')
        for i in range(predictions.shape[0]):
            predValue = predictions[i][0]
            print('  {0:s}:\033[1m\n   Predicted value = {1:.2f}\033[0m\n'.format(fileName[i],predValue))
            summaryFile = np.vstack((summaryFile,[fileName[i],predValue,'']))
        print('  ========================================================\n')

    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        
        summaryFile = np.array([['SpectraKeras_CNN','Classifier','',''],['File name','Name','Predicted Class','Probability']])
        print('\n  ========================================================')
        print('  \033[1m CNN - Classifier\033[0m - Prediction')
        print('  ========================================================')
        indPredProb = 0
        for i in range(predictions.shape[0]):
            pred_class = np.argmax(predictions[i])
            if dP.useTFlitePred:
                predProb = round(100*predictions[i][pred_class]/255,2)
            else:
                predProb = round(100*predictions[i][pred_class],2)
            rosterPred = np.atleast_1d(predictions[i][0]).nonzero()[0]

            if pred_class.size >0:
                predValue = le.inverse_transform(pred_class)[0]
                print('  {0:s}:\033[1m\n   {1:s} \033[0m(Class: {2:d}, probability = {3:.2f}%)\n'.format(fileName[i],getMineral(dP.table_names, int(predValue)), int(predValue), predProb))
            else:
                predValue = 0
                print('  {0:s}:\033[1m\n   No predicted value (probability = {1:.2f}%)\033[0m\n'.format(fileName[i],predProb))
            if predProb > dP.predProbThreshold:
                indPredProb += 1
            summaryFile = np.vstack((summaryFile,[fileName[i], getMineral(dP.table_names, int(predValue)), predValue,predProb]))
        print('  ========================================================\n')
        print(" Predictions with probability > {0:.2f}:  {1:.2f}%\n".format(dP.predProbThreshold, indPredProb*100/predictions.shape[0]))
        
    import pandas as pd
    df = pd.DataFrame(summaryFile)
    df.to_csv(dP.summaryFileName, index=False, header=False)
    print(" Prediction summary saved in:",dP.summaryFileName,"\n")

#************************************
# Accuracy determination
#************************************
def accDeterm(testFile):
    dP = Conf()
    model = loadModel(dP)
    En, A, Cl = readLearnFile(testFile, dP)
    predictions = np.zeros((0,0))
    fileName = []
    print("\n  Number of spectra in testing file:",A.shape[0])

    for row in A:
        R = np.array([row])
        try:
            predictions = np.vstack((predictions,getPredictions(R, model, dP)[0].flatten()))
        except:
            predictions = np.array([getPredictions(R, model,dP)[0].flatten()])

    if dP.regressor:
        print("\n  Accuracy determination is not defined in regression. Exiting.\n")
        return
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        summaryFile = np.array([['SpectraKeras_CNN','Classifier',''],['Real Class','Predicted Class', 'Probability']])

        successPred = 0

        for i in range(predictions.shape[0]):
            pred_class = np.argmax(predictions[i])
            if dP.useTFlitePred:
                predProb = round(100*predictions[i][pred_class]/255,2)
            else:
                predProb = round(100*predictions[i][pred_class],2)
            rosterPred = np.where(predictions[i][0]>0.1)[0]

            if pred_class.size >0:
                predValue = le.inverse_transform(pred_class)[0]
            else:
                predValue = 0
            if Cl[i] == predValue:
                successPred+=1

            summaryFile = np.vstack((summaryFile,[Cl[i], predValue, predProb]))

    print("\n\033[1m  Overall average accuracy: {0:.2f}% \033[0m".format(successPred*100/Cl.shape[0]))
    summaryFile[0,2] = "Av Acc: {0:.2f}%".format(successPred*100/Cl.shape[0])

    import pandas as pd
    df = pd.DataFrame(summaryFile)
    df.to_csv(dP.summaryAccFileName, index=False, header=False)
    print("\n  Prediction summary saved in:",dP.summaryAccFileName,"\n")

#****************************************************
# Convert model to quantized TFlite
#****************************************************
def convertTflite(learnFile):
    dP = Conf()
    dP.useTFlitePred = False
    dP.TFliteRuntime = False
    dP.runCoralEdge = False
    learnFileRoot = os.path.splitext(learnFile)[0]
    En, A, Cl = readLearnFile(learnFile, dP)
    model = loadModel(dP)
    makeQuantizedTFmodel(A, dP)

#************************************
# Print NN Info
#************************************
def printParam():
    dP = Conf()
    print('\n  ================================================')
    print('  \033[1m MLP\033[0m - Parameters')
    print('  ================================================')
    print('  Optimizer:','Adam',
                '\n  Hidden layers:', dP.HL,
                '\n  Activation function:','relu',
                '\n  L2:',dP.l2,
                '\n  Dropout:', dP.drop,
                '\n  Learning rate:', dP.l_rate,
                '\n  Learning decay rate:', dP.l_rdecay)
    if dP.fullSizeBatch:
        print('  Batch size: full')
    else:
        print('  Batch size:', dP.batch_size)
    print('  Number of labels:', dP.numLabels)
    print('  Epochs:',dP.epochs)
    print('  Stop at Best Model based on validation:', dP.stopAtBest)
    print('  Save Best Model based on validation:', dP.saveBestModel)
    if dP.regressor:
        print('  Metric for Best Regression Model:', dP.metricBestModelR)
    else:
        print('  Metric for Best Classifier Model:', dP.metricBestModelC)
    #print('  ================================================\n')

#************************************
# Lists the program usage
#************************************
def usage():
    print('\n Usage:\n')
    print(' Train (Random cross validation):')
    print('  python3 SpectraKeras_MLP.py -t <learningFile>\n')
    print(' Train (with external validation):')
    print('  python3 SpectraKeras_MLP.py -t <learningFile> <validationFile>\n')
    print(' Predict:')
    print('  python3 SpectraKeras_MLP.py -p <testFile>\n')
    print(' Batch predict:')
    print('  python3 SpectraKeras_MLP.py -b <folder>\n')
    print(' Convert model to quantized tflite:')
    print('  python3 SpectraKeras_MLP.py -l <learningFile>\n')
    print(' Determine accuracy using h5 testing file with spectra:')
    print('  python3 SpectraKeras_MLP.py -a <testFile>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
