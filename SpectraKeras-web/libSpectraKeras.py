# -*- coding: utf-8 -*-
'''
**********************************************
* libSpectraKeas - Library for SpectraKeras
* v2024.10.10.1
* Uses: TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
**********************************************
'''
import numpy as np
import scipy
import os.path, pickle, h5py

#************************************
# Open Learning Data
#************************************
def readLearnFile(learnFile, dP):
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
# Open Testing Data
#************************************
def readTestFile(testFile, En, dP):
    try:
        with open(testFile, 'r') as f:
            #print('\n  Opening sample data for prediction:\n  ',testFile)
            Rtot = np.loadtxt(f, unpack =True)
        R = preProcess(Rtot, En, dP)
    except:
        print("\033[1m\n File not found or corrupt\033[0m\n")
        return 0, False
    return R, True

#****************************************************
# Check Energy Range and convert to fit training set
#****************************************************
def preProcess(Rtot, En, dP):
    R = np.array([Rtot[1,:]])
    Rx = np.array([Rtot[0,:]])

    if dP.normalize:
        norm = Normalizer()
        R = norm.transform_single(R)

    if(R.shape[1] is not len(En)):
        print('  Rescaling x-axis from',str(R.shape[1]),'to',str(len(En)))
        R = np.interp(En, Rx[0], R[0])
        R = R.reshape(1,-1)
    return R

#************************************
# Load saved models
#************************************
def loadModel(dP):
    if dP.TFliteRuntime:
        import tflite_runtime.interpreter as tflite
        # model here is intended as interpreter
        if dP.runCoralEdge:
            #print(" Running on Coral Edge TPU")
            try:
                model = tflite.Interpreter(model_path=os.path.splitext(dP.model_name)[0]+'_edgetpu.tflite',
                    experimental_delegates=[tflite.load_delegate(dP.edgeTPUSharedLib,{})])
            except:
                print(" Coral Edge TPU not found. Please make sure it's connected and Tflite-runtime matches the TF version that is installled.")
        else:
            model = tflite.Interpreter(model_path=os.path.splitext(dP.model_name)[0]+'.tflite')
        model.allocate_tensors()
    else:
        getTFVersion(dP)
        import tensorflow as tf
        if checkTFVersion("2.16.0"):
            import tensorflow.keras as keras
        else:
            if dP.kerasVersion == 2:
                import tf_keras as keras
            else:
                import keras
        if dP.useTFlitePred:
            # model here is intended as interpreter
            model = tf.lite.Interpreter(model_path=os.path.splitext(dP.model_name)[0]+'.tflite')
            model.allocate_tensors()
        else:
            if dP.kerasVersion == 2:
                model = keras.models.load_model(dP.model_name)
            else:
                model = keras.saving.load_model(dP.model_name)
    print("  Model name:", dP.model_name)
    return model

#************************************
# Make prediction based on framework
#************************************
def getPredictions(R, model, dP):
    if dP.useTFlitePred:
        interpreter = model  #needed to keep consistency with documentation
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(R*255, dtype=np.uint8) # Disable this for TF1.x
        #input_data = np.array(R, dtype=np.float32)  # Enable this for TF2.x (not compatible with on EdgeTPU)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        predictions = interpreter.get_tensor(output_details[0]['index'])
    else:
        predictions = model.predict(R)
        
    probabilities = scipy.special.softmax(predictions.astype('double'))
    return predictions, probabilities

#************************************
### Create Quantized tflite model
#************************************
def makeQuantizedTFmodel(A, dP):
    import tensorflow as tf
 
    print("\n  =========================================================")
    print("   Creating\033[1m quantized TensorFlowLite Model \033[0m")
    print("  =========================================================\n")

    A2 = tf.cast(A, tf.float32)
    A = tf.data.Dataset.from_tensor_slices((A2)).batch(1)

    def representative_dataset_gen():
        for input_value in A.take(100):
            yield[input_value]

    if dP.kerasVersion == 2:
        if checkTFVersion("2.16.0"):
            import tensorflow.keras as keras
        else:
            import tf_keras as keras
        model = keras.models.load_model(dP.model_name)
    else:
        # Previous method, TF <= 2.16.2
        #import keras
        #model = keras.layers.TFSMLayer(dP.model_name, call_endpoint='serve')
        #converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # New method
        #model = tf.saved_model.load(dP.model_name)
        #concrete_func = model.signatures['serving_default']
        #converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        # New method 2:
        import keras
        model = keras.saving.load_model(dP.model_name)
        
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()

    with open(os.path.splitext(dP.model_name)[0]+'.tflite', 'wb') as o:
        o.write(tflite_quant_model)

#************************************
# Plot Weights
#************************************
def plotWeights(dP, En, A, model, type):
    import matplotlib.pyplot as plt
    if checkTFVersion("2.16.0"):
        import tensorflow as tf
        import tensorflow.keras as keras
    else:
        #import tf_keras as keras
        import keras
    plotFileName = "model_" + type + "_weights" + ".png"
    plt.figure(tight_layout=True)
    #plotInd = 511
    plotInd = (len(dP.HL)+2)*100+11
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w_layer = layer.get_weights()[0]
            ax = plt.subplot(plotInd)
            newX = np.arange(En[0], En[-1], (En[-1]-En[0])/w_layer.shape[0])
            plt.plot(En, np.interp(En, newX, w_layer[:,0]), label=layer.get_config()['name'])
            plt.legend(loc="upper right")
            plt.setp(ax.get_xticklabels(), visible=False)
            plotInd +=1
            print(" Preparing weigths for layer:",layer.name)

    ax1 = plt.subplot(plotInd)
    ax1.plot(En, A[0], label='Sample data')

    plt.xlabel("Raman shift [1/cm]")
    plt.legend(loc="upper right")
    plt.savefig(plotFileName, dpi = 160, format = 'png')  # Save plot
    print(" Saving weights plots in:", plotFileName,"\n")

#************************************
# Get TensorFlow Version
#************************************
def getTFVersion(dP):
    import tensorflow as tf
    if checkTFVersion("2.16.0"):
        import tensorflow.keras as keras
        kv = "- Keras v. " + keras.__version__
    else:
        if dP.kerasVersion == 2:
            import tf_keras as keras
            kv = "- tf_keras v. " + keras.__version__
        else:
            import keras
            kv = "- Keras v. " + keras.__version__
    from packaging import version
    if dP.useTFlitePred:
        print("\n TensorFlow (Lite) v.",tf.version.VERSION,kv, "\n")
    else:
        print("\n TensorFlow v.",tf.version.VERSION,kv, "\n" )
        
def checkTFVersion(vers):
    import tensorflow as tf
    from packaging import version
    v = version.parse(tf.__version__)
    return v < version.parse(vers)

#************************************
# Normalizer
#************************************
class Normalizer(object):
    def __init__(self):
        self.YnormTo = 1
        print("  Normalizing spectra between 0 and 1")

    def transform_matrix(self,y):
        yn = np.copy(y)
        for i in range(0,y.shape[0]):
            if np.amax(y[i,:]) - np.amin(y[i,:]) == 0:
                pass
            else:
                yn[i,:] = np.multiply(y[i,:] - np.amin(y[i,:]),
                    self.YnormTo/(np.amax(y[i,:]) - np.amin(y[i,:])))
        return yn

    def transform_single(self,y):
        yn = np.copy(y)
        yn = np.multiply(y - np.amin(y),
                self.YnormTo/(np.amax(y) - np.amin(y)))
        return yn

    def save(self, name):
        with open(name, 'ab') as f:
            pickle.dump(self, f)

#************************************
# Normalize Label
#************************************
class NormalizeLabel(object):
    def __init__(self, M, dP):
        self.M = M
        self.normalizeLabel = dP.normalizeLabel
        self.useGeneralNormLabel = dP.useGeneralNormLabel
        self.useCustomRound = dP.useCustomRound
        self.minGeneralLabel = dP.minGeneralLabel
        self.maxGeneralLabel = dP.maxGeneralLabel
        self.YnormTo = dP.YnormTo
        self.stepNormLabel = dP.stepNormLabel

        self.data = np.arange(0,1,self.stepNormLabel)
        self.min = np.zeros([self.M.shape[1]])
        self.max = np.zeros([self.M.shape[1]])

        if self.normalizeLabel:
            if self.useGeneralNormLabel:
                self.min[0] = dP.minGeneralLabel
                self.max[0] = dP.maxGeneralLabel
            else:
                self.min[0] = np.amin(self.M[1:,0])
                self.max[0] = np.amax(self.M[1:,0])

        for i in range(1,M.shape[1]):
            self.min[i] = np.amin(self.M[1:,i])
            self.max[i] = np.amax(self.M[1:,i])

    def transform_matrix(self,y):
        Mn = np.copy(y)
        if self.normalizeLabel:
            Mn[1:,0] = np.multiply(y[1:,0] - self.min[0],
                self.YnormTo/(self.max[0] - self.min[0]))
            if self.useCustomRound:
                customData = CustomRound(self.data)
                for i in range(1,y.shape[0]):
                    Mn[i,0] = customData(Mn[i,0])

        for i in range(1,y.shape[1]):
            Mn[1:,i] = np.multiply(y[1:,i] - self.min[i],
                self.YnormTo/(self.max[i] - self.min[i]))
        return Mn

    def transform_valid(self,V):
        Vn = np.copy(V)
        for i in range(0,V.shape[0]):
            Vn[i,1] = np.multiply(V[i,1] - self.min[i+1],
                self.YnormTo/(self.max[i+1] - self.min[i+1]))
        return Vn

    def transform_inverse_single(self,v):
        vn = self.min[0] + v*(self.max[0] - self.min[0])/self.YnormTo
        return vn

    def save(self, name):
        with open(name, 'ab') as f:
            pickle.dump(self, f)

#************************************
# CustomRound
#************************************
class CustomRound:
    def __init__(self,iterable):
        self.data = sorted(iterable)

    def __call__(self,x):
        data = self.data
        ndata = len(data)
        from bisect import bisect_left
        idx = bisect_left(data,x)
        if idx <= 0:
            return data[0]
        elif idx >= ndata:
            return data[ndata-1]
        x0 = data[idx-1]
        x1 = data[idx]
        if abs(x-x0) < abs(x-x1):
            return x0
        return x1

#************************************
# MultiClassReductor
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
        return [self.totalClass[int(a)]]

    def classes_(self):
        return self.totalClass
