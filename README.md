# SpectralMachine
Machine learning software for rapid spectral analysis. While Raman spectra were the initilal focus, SpectralMachine is flexible to be applied for classification using any spectra (from XRD, FTIR and beyond). The latest and supporrted software within SpectralMachine is SpectraKeras. The previous generation (SpectraLearnPredict) is no longer developed. 

**SpectraKeras**
- Currently supported ML architectures:
   - DNNClassifier (TensorFlow, TensorFlow-Lite)
   - Convolutional Neural Networks (TensorFlow-Lite)
- Required libraries for prediction:
   - tensorflow (v.2.3 and higher)
   - Optional: tensorflow-lite (v.2.3 and higher)
   - Optional: [tensorflow-lite runtime](https://www.tensorflow.org/lite/guide/python) 
   - Optional: tensorflow-lite runtime with [Coral EdgeTPU](https://coral.ai/docs/accelerator/get-started/)

**Previous version: SpectralLearnPredict**
- This is deprecated and no longer developed.
 - Deep Neural Networks:
   - multi-layer perceptron (MLP) (L-BFGS Optimizer strongly recommended)
   - DNNClassifier (TensorFlow and keras)
   - Convolutional Neural Networks (Under development - via keras)
 - Support Vector Machine - SVM
 - TensorFlow (basic implementation)
 - Additional multivariate analysis
   - K-Means
   - Principal component analysis

Credits and References
==================
If you use SpectralMachine or SpectraKeras, we request that you reference the papers/resources on which SpectralMachine is based:
- N Ferralis, SpectralMachine and SpectraKeras (2017), https://github.com/feranick/SpectralMachine
- N. Ferralis, R.E. Sumons, Automated Mineral and Geochemical Classification from Spectroscopy Data using Machine Learning, AGU Fall Meeting 2018, Washington, DC, December 2018, https://agu.confex.com/agu/fm18/prelim.cgi/Paper/369458 
- N. Ferralis, Automated Materials Classification from Spectroscopy/Diffraction through Deep Neural Networks, MRS Fall Meeting 2018, Boston, MA, November 2018, https://mrsfall2018.zerista.com/event/member/533306

Installation
=============

This software requires Python (3.6 or higher). It has been tested with Python 3.6 or higher which is the recommended platform. It is not compatible with python 2.x. Additional required packages:

    numpy
    scikit-learn (>=0.18)
    matplotlib
    pandas
    pydot
    graphviz
    h5py
    tensorflow (>=2.3, not compatible with TF 1.x)
    
In addition, these packages may be needed depending on your platform (via ```apt-get``` in debian/ubuntu or ```port``` in OSX):
    
    python3-tk
    graphviz

These are found in Unix based systems using common repositories (apt-get for Debian/Ubuntu Linux, or MacPorts for MacOS). More details in the [scikit-learn installation page](http://scikit-learn.org/stable/install.html).

[TensorFlow](https://www.tensorflow.org) is needed only if flag is activated. Instructions for Linux and MacOS can be found in [TensorFlow installation page](https://www.tensorflow.org/install/). Pip installation is the easiest way to get going. Tested with TensorFlow v.1.15+. TensorFlow 2.x (2.3 or higher preferred) is the currently sipported release. 

Prediction can be carried out using the regular tensorflow, or using [tensorflow-lite](https://www.tensorflow.org/lite/) for [quantized models](https://www.tensorflow.org/lite/performance/post_training_quantization). Loading times of tflite (direct or via [tflite-runtime](https://www.tensorflow.org/lite/guide/python)) are significantly faster than tensorflow with minimal loss in accuracy. SpectraKeras provides an option to convert tensorflow models to quantized tflite models. TFlite models have been tested in Linux x86-64, arm7 (including Raspberry Pi3) and aarm64, MacOS, Windows. For using quantized model (specifically when deployed on Coral EdgeTPU), TF 2.3 or higher is recommended. 

Usage (SpectraKeras)
===================

## SpectraKeras_CNN

Train (Random cross validation):

    python3 SpectraKeras_CNN.py -t <learningFile>
    
Train (with external validation):

    python3 SpectraKeras_CNN.py -t <learningFile> <validationFile>
    
Predict:

    python3 SpectraKeras_CNN.py -p <testFile>
    
Batch predict:

    python3 SpectraKeras_CNN.py -b <folder>

Display Neural Netwrok Configuration:
    
    python3 SpectraKeras_CNN.py -n <learningFile>
    
Convert model to quantized tflite:
    
    python3 SpectraKeras_CNN.py -l <learningFile>
    
Determine accuracy using h5 testing file with spectra:

    python3 SpectraKeras_CNN.py -a <testFile>
    
## SpectraKeras_MLP

Train (Random cross validation):

    python3 SpectraKeras_MLP.py -t <learningFile>
        
Train (with external validation):

    python3 SpectraKeras_MLP.py -t <learningFile> <validationFile>
        
Predict:

    python3 SpectraKeras_MLP.py -p <testFile>
        
Batch predict:
        
    python3 SpectraKeras_MLP.py -b <folder>
        
Convert model to quantized tflite:
        
    python3 SpectraKeras_MLP.py -l <learningFile>
        
Determine accuracy using h5 testing file with spectra:

    python3 SpectraKeras_MLP.py -a <testFile>

Formatting input file for training
========================
The main idea behind the software is to train classification or regression models from plain spectra (which can be Raman, but they can be any spectra or diffraction profiles, as long as the model is consistent), rather than from manually selected features (such as bands, peaks, widths, etc). So, suppose one has training files similar to this, where the first column is the Raman shiift, the second is intensity:

1000  123
1001  140
1002  180
1003  150
...

Let's say this file correspond to label 1, and now one has a collection of files that will be used for training each with its own label, the input file will look like this:

0  1000  1001  1002  1003 ...
lab1  123 140  180  150  ...
lab2 ... ... ... ... ...

Essentially each line in the input file corresponds to a training file with its label. during training the model will learn (either through a simple deep MLP network using SpectraKeras_MLP, or through a Convolutional Network using SpectraKeras_CNN, which is recommended) to extract features needed for prediction. Note that all spectra needs to have the same Raman shifts max min and step.

Of course it is not expected that the user manually compiles the training file from a possibly large collection of spectra. For that, GenericDataMaker.py is available in the "Utilities" folder, that can be used to automatically create such files. Basically you can run from the folder where you have your spectra:

python3 GenericDataMaker.py <learnfile> <enInitial> <enFinal> <enStep>

The script will interpolate each spectra within the Raman shifts parameters you set above. Note that there are some basic configuration that you may need to change in the GenericDataMakerp.py for your case (such as delimiter between data, extension of the files, etc).

One can use the same to create a validation file, or you can use other scripts also provided to split the training set into training+validation. That can be done randomly within SpectraKeras, but the split will be different every time you run it.

Once models are trained trained, prediction on individual files can be made using simply formatted ASCII files (like in the example above).

Training data
=============
We do not provide advanced training sets, some of which can be found online. We only provide a simple Raman dataset mainly for testing purposes: it is loosely based on 633nm data from Ferralis et al. [Carbon 108 (2016) 440](http://dx.doi.org/10.1016/j.carbon.2016.07.039).


Data augmentation
==================
Using the provided training data set as is, accuracy is low. For a single training run using a random 30% of the training set for 100 times, the accuracy is about 32%:

./SpectraLearnPredict.py -t Training/20170227a/Training_kerogen_633nm_HC_20170227a.txt 1

Repeating the training few more times (5, for example) marginally increases the accuracy to 35.7% and it is fully converged. This is expected given the small dataset.

Increasing the number of spectra in the actual dataset can be done by accounting for noise. Using the AddNoisyData.py utility, the esisting training set is taken, and random variations in intensity at each energy point are added within a given offset. This virtually changes the overall spectra, without changing its overall trend in relation to the classifier (H:C). This allows for the preservation of the classifier for a given spectra, but it also increases the number of available spectra. This method is obviously a workaround, but it allows accuracy to be substantially increased. Furthermore, it lends a model better suited at dealing with noisy data. 

To recreate, let's start with adding noisy spectra to the training set. For example, let's add 5 replicated spectra with random noise added with an offset of 0.02 (intensity is normalized to the range [0,1])

AddNoisyData.py Training/20170227a/Training_kerogen_633nm_HC_20170227a.txt 5 0.02

Accuracy is increased to 80.4% with a single training run (100 times 30% of the dataset). 2 iterations increase accuracy to 95.8% and a third increased to 100%. (The same can be achieved by running 30% of the dataset 300 times).

One can optimize/minimize the number of spectra with added noise. Adding only 2 data-sets with noise offset at 0.02 converges the accuracy to about 94.6%. 

**One final word of caution**: Increasing the number of statistically independent available spectra for training is recommended over adding noisy data. 

More on Machine Learning tools used
====================================

- [Neural Networks](http://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [Deep Neural Networks - Multilayer Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Deep Neural Networks - TensorFlow](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier)
- [TensorFlow](https://www.tensorflow.org)
- [Support Vector Classification](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [K-Means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Principal Component Analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

Known Issues
==================
Starting from version 20210513a, models are saved with the `.h5` extension, and are expected to have that extension. Previously, models were saved with the non-standard `.hd5` extension. If you have previously trained models saved in `.hd5`, just rename the extension as `.h5`. No change in functionality besides the change in extension.
