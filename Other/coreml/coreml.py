#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************
* SpectraKeras_CNN to Core ML Converter
* v2024.11.19.1
* Uses: coremltools
* By: Nicola Ferralis <feranick@hotmail.com>
**********************************************
'''
print(__doc__)

import pickle, sys
import coremltools as ct
from libSpectraKeras import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def coremltools():
    main()
    
class Conf():
    kerasVersion = 2
    
#************************************
# Main
#************************************
def main():
    dP = Conf()
    with open("model_spectral_range.pkl", "rb") as f:
        EnN = pickle.load(f)
        
    with open("model_le.pkl", "rb") as f:
        le = pickle.load(f)
    
    Cl = le.totalClass
    
    keras_model_name = "model_classifier_CNN.h5"
    
    if dP.kerasVersion == 2:
        import tf_keras as keras
        keras_model = keras.models.load_model(keras_model_name)
    else:
        import keras
        keras_model = keras.saving.load_model(keras_model_name)
    
    # Define the input type as image,
    # set pre-processing parameters to normalize the image
    # to have its values in the interval [-1,1]
    # as expected by the mobilenet model
    image_input = ct.ImageType(shape=(1, 1, EnN.shape[0], 1,),
                           bias=[-1,-1,-1], scale=1)

    # set class labels
    classifier_config = ct.ClassifierConfig(Cl)

    # Convert the model using the Unified Conversion API to an ML Program
    model = ct.convert(
        keras_model,
        inputs=[image_input],
        classifier_config=classifier_config,
        source="tensorflow",
    )

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

    En = M[0,1:]
    A = M[1:,1:]

    if dP.normalize:
        norm = Normalizer()
        A = norm.transform_matrix(A)

    Cl = M[1:,0]

    return En, A, Cl


#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
