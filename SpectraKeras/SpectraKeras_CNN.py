#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
* SpectraKeras - CNN
* 20181012a
* Uses: Keras, TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, getopt, time, configparser, pickle, h5py

#***************************************************
''' This is needed for installation through pip '''
#***************************************************
def SpectraKeras_CNN():
    main()

#************************************
''' Parameters '''
#************************************
class Conf():
    def __init__(self):
        confFileName = "SpectraKeras_CNN.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
            
    def SKDef(self):
        self.conf['Parameters'] = {
            'l_rate' : 0.001,
            'l_rdecay' : 1e-4,
            'CL_filter' : [1],
            'CL_size' : [10],
            'max_pooling' : 20,
            'HL' : [40,70],
            'drop' : 0,
            'l2' : 1e-4,
            'epochs' : 100,
            'cv_split' : 0.01,
            'fullSizeBatch' : True,
            'batch_size' : 256,
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
        
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.CL_filter = eval(self.SKDef['CL_filter'])
            self.CL_size = eval(self.SKDef['CL_size'])
            self.max_pooling = eval(self.SKDef['max_pooling'])
            
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
''' Main '''
#************************************
def main():
    start_time = time.clock()
    dP = Conf()
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "tph:", ["train", "predict", "help"])
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

    total_time = time.clock() - start_time
    print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

#************************************
''' Training '''
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

    #from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

    En, A, Cl = readLearnFile(learnFile)
    learnFileRoot = os.path.splitext(learnFile)[0]
    
    if testFile != None:
        En_test, A_test, Cl_test = readLearnFile(testFile)
        totA = np.vstack((A, A_test))
        totCl = np.append(Cl, Cl_test)
    else:
        totA = A
        totCl = Cl

    tb_directory = "keras_CNN"
    model_directory = "."
    model_name = model_directory+"/keras_CNN_model.hd5"
    model_le = model_directory+"/keras_le.pkl"
    
    #************************************
    ''' Label Encoding '''
    #************************************
    
    # sklearn preprocessing is only for single labels
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    totCl2 = le.fit_transform(totCl)
    Cl2 = le.transform(Cl)
    '''
    le = MultiClassReductor()
    le.fit(np.unique(Cl, axis=0))
    Cl2 = le.transform(Cl)
    '''

    print("  Number of learning labels:",dP.numLabels)
    print("  Number unique classes (training): ", np.unique(Cl).size)

    if testFile != None:
        Cl2_test = le.transform(Cl_test)
        print("  Number unique classes (validation):", np.unique(Cl_test).size)
        print("  Number unique classes (total): ", np.unique(totCl).size)

    print("  Total number of points per data:",En.size)

    print("\n  Label Encoder saved in:", model_le,"\n")
    with open(model_le, 'ab') as f:
        f.write(pickle.dumps(le))

    #************************************
    ''' Training '''
    #************************************
    #totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
    Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(totCl).size+1)
    if testFile != None:
        Cl2_test = keras.utils.to_categorical(Cl2_test, num_classes=np.unique(totCl).size+1)

    if dP.fullSizeBatch == True:
        dP.batch_size = A.shape[0]

    # Format spectra as images for loading
    listmatrix = []
    for i in range(A.shape[0]):
        spectra = np.dstack([np.ones(En.shape[0]), En])
        spectra = np.dstack([spectra, A[i]])
        listmatrix.append(spectra)

    x_train = np.stack(listmatrix, axis=0)
    y_train = Cl2
    
    ### Build model
    model = keras.models.Sequential()

    for i in range(len(dP.CL_filter)):
        model.add(keras.layers.Conv2D(dP.CL_filter[i], (1, dP.CL_size[i]),
            activation='relu',
            input_shape=spectra.shape))
    try:
        model.add(keras.layers.MaxPooling2D(pool_size=(1, dP.max_pooling)))
    except:
        dP.max_pooling -= dP.CL_size[-1] - 1
        print(" Rescaled pool size: ", dP.max_pooling, "\n")
        model.add(keras.layers.MaxPooling2D(pool_size=(1, dP.max_pooling)))

    model.add(keras.layers.Dropout(dP.drop))
    model.add(keras.layers.Flatten())

    for i in range(len(dP.HL)):
        model.add(keras.layers.Dense(dP.HL[i],
            activation = 'relu',
            input_dim=A.shape[1],
            kernel_regularizer=keras.regularizers.l2(dP.l2)))
        model.add(keras.layers.Dropout(dP.drop))
    model.add(keras.layers.Dense(np.unique(totCl).size+1, activation = 'softmax'))

    #optim = opt.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    optim = keras.optimizers.Adam(lr=dP.l_rate, beta_1=0.9,
                    beta_2=0.999, epsilon=1e-08,
                    decay=dP.l_rdecay,
                    amsgrad=False)

    model.compile(loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy'])

    tbLog = keras.callbacks.TensorBoard(log_dir=tb_directory, histogram_freq=120,
            batch_size=dP.batch_size,
            write_graph=True, write_grads=True, write_images=True)
    tbLogs = [tbLog]
    if testFile != None:
        log = model.fit(x_train, y_train,
            epochs=dP.epochs,
            batch_size=dP.batch_size,
            callbacks = tbLogs,
            verbose=2,
	        validation_data=(A_test, Cl2_test))
    else:
        log = model.fit(x_train, y_train,
            epochs=dP.epochs,
            batch_size=dP.batch_size,
            callbacks = tbLogs,
            verbose=2,
            validation_split=dP.cv_split)

    accuracy = np.asarray(log.history['acc'])
    loss = np.asarray(log.history['loss'])
    val_loss = np.asarray(log.history['val_loss'])
    val_acc = np.asarray(log.history['val_acc'])

    #score = model.evaluate(A_test, Cl2_test, batch_size=A.shape[1])
    model.save(model_name)
    keras.utils.plot_model(model, to_file=model_directory+'/keras_CNN_model.png', show_shapes=True)
    
    print('\n  =============================================')
    print('  \033[1mKeras CNN\033[0m - Model Configuration')
    print('  =============================================')
    #for conf in model.get_config():
    #    print(conf,"\n")
    print("  Training set file:",learnFile)
    print("  Data size:", A.shape,"\n")
    print("  Number of learning labels:",dP.numLabels)
    print("  Number unique classes (training): ", np.unique(Cl).size)
    if testFile != None:
        Cl2_test = le.transform(Cl_test)
        print("  Number unique classes (validation):", np.unique(Cl_test).size)
        print("  Number unique classes (total): ", np.unique(totCl).size)
    print("  Total number of points per data:",En.size)
    printParam()

    print('\n  ==========================================')
    print('  \033[1mKeras CNN\033[0m - Training Summary')
    print('  ==========================================')
    print("\n  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(accuracy), 100*np.amax(accuracy)))
    print("  Loss - Average: {0:.4f}; Min: {1:.4f}".format(np.average(loss), np.amin(loss)))
    print('\n\n  ==========================================')
    print('  \033[1mKeras CNN\033[0m - Validation Summary')
    print('  ==========================================')
    print("\n  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(val_acc), 100*np.amax(val_acc)))
    print("  Loss - Average: {0:.4f}; Min: {1:.4f}\n".format(np.average(val_loss), np.amin(val_loss)))
    #print("\n  Validation - Loss: {0:.2f}; accuracy: {1:.2f}%".format(score[0], 100*score[1]))
    print('  =========================================\n')

    if dP.plotWeightsFlag == True:
        plotWeights(En, A, model)

#************************************
''' Predict '''
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
    
    le = pickle.loads(open("keras_le.pkl", "rb").read())
    model = keras.models.load_model("keras_CNN_model.hd5")
    predictions = model.predict(R, verbose=1)
    pred_class = np.argmax(predictions)
    
    if pred_class.size >0:
        predValue = le.inverse_transform([pred_class])[0]
    else:
        predValue = 0

    predProb = round(100*predictions[0][pred_class],2)
    rosterPred = np.where(predictions[0]>0.1)[0]

    if dP.numLabels == 1:
        print('\033[1m' + '\n Predicted value (Keras) = ' + str(predValue) +
          '  (probability = ' + str(predProb) + '%)\033[0m\n')
    else:
        print('\n ==========================================')
        print('\033[1m' + ' Predicted value \033[0m(probability = ' + str(predProb) + '%)')
        print(' ==========================================\n')

#************************************
''' Open Learning Data '''
#************************************
def readLearnFile(learnFile):
    print("\n Opening learning file: "+learnFile+"\n")
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
        print("\033[1m" + " Learning file not found \n" + "\033[0m")
        return

    dP = Conf()
    En = M[0,dP.numLabels:]
    A = M[1:,dP.numLabels:]
    if dP.numLabels == 1:
        Cl = M[1:,0]
    else:
        Cl = M[1:,[0,dP.numLabels-1]]

    return En, A, Cl

#************************************
''' Print NN Info '''
#************************************
def printParam():
    dP = Conf()
    print('\n  ================================================')
    print('  \033[1mKeras CNN\033[0m - Parameters')
    print('  ================================================')
    print('  Optimizer:','Adam',
                '\n  Convolutional layers:', dP.CL_filter,
                '\n  Convolutional layers size:', dP.CL_size,
                '\n  Max Pooling:', dP.max_pooling,
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
''' Open Learning Data '''
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
    plt.savefig('keras_CNN_weights' + '.png', dpi = 160, format = 'png')  # Save plot

#************************************
''' MultiClassReductor '''
#************************************
class MultiClassReductor():
    def __self__(self):
        self.name = name
    
    def fit(self,tc):
        self.totalClass = tc.tolist()
    
    def transform(self,y):
        Cl = np.zeros(y.shape[0])
        for j in range(len(y)):
            Cl[j] = self.totalClass.index(np.array(y[j]).tolist())
        return Cl
    
    def inverse_transform(self,a):
        return [self.totalClass[int(a[0])]]

    def classes_(self):
        return self.totalClass

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print(' Train (Random cross validation):')
    print('  python3 SpectraKeras_CNN.py -t <learningFile>\n')
    print(' Train (with external validation):')
    print('  python3 SpectraKeras_CNN.py -t <learningFile> <validationFile>\n')
    print(' Predict:')
    print('  python3 SpectraKeras_CNN.py -p <testFile>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
