#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************************
* Convert TF Keras V2 models into TF.Lite
* v2024.02.28.1
* Uses: TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)
import sys, os.path, h5py

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3.py ConvertToTFLite <Model in Keras format>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        print(' Converting',sys.argv[1],'to TF.Lite...\n')
        convertModelToTFLite(sys.argv[1])

#************************************
# Convert TF Model to TF.Lite
#************************************
def convertModelToTFLite(model_file):
    import tensorflow as tf
    if checkTFVersion("2.15.99"):
        import tensorflow.keras as keras
    else:
        import tf_keras as keras
    convFile = os.path.splitext(model_file)[0]+'.tflite'
    try:
        model = keras.models.load_model(model_file)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)    # TensorFlow 2.15 and earlier
    except RuntimeError as err:
        print(err);
        #converter = tf.lite.TFLiteConverter.from_keras_model_file(model_file)  # TensorFlow 1.x
        
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)
    open(convFile, "wb").write(tflite_model)
    print(' Converted model saved to:',convFile,'\n')

#************************************
# Get TensorFlow Version
#************************************
def checkTFVersion(vers):
    import tensorflow as tf
    from packaging import version
    v = version.parse(tf.version.VERSION)
    return v < version.parse(vers)

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())


