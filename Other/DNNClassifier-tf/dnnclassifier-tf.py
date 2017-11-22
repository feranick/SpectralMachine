import os

import six.moves.urllib.request as request
import tensorflow as tf
import numpy as np

import tensorflow.contrib.learn as skflow
from sklearn import preprocessing
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

PATH = ""
FILE_TRAIN = "Training_kerogen_633nm_HC_20170524a.txt"
#FILE_TRAIN = PATH+"AAA_ram_ex-unor_train.txt"
FILE_TEST = "AAA_ram_ex-unor_test.txt"

def get_feature_names(learnFile):
    try:
        with open(learnFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
        feature_names = np.char.mod('%s',M[0,1:][0:])
        return feature_names
    except:
        print('\033[1m' + ' Learning file not found \n' + '\033[0m')
        return

feature_names = get_feature_names(FILE_TRAIN)
print(feature_names.shape)

def my_input_fn(learnFile,file_path, perform_shuffle=False, repeat_count=1):

    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[i] for i in range(len(feature_names)+1)])
        label = parsed_line[0]  # Last element is the label
        print("\n\nlabel\n",label,"\n\n\n")
        #del parsed_line[0]  # Delete last element
        features = parsed_line[1:]  # Everything but last elements are the features
        print("\n\nfeatures\n",len(features),"\n\n",features,"\n\n")

        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(learnFile)  # Read text file
               .skip(1)  # Skip header row
               .map(decode_csv))

    print(dataset)
    
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(32)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

next_batch = my_input_fn(FILE_TRAIN, True)  # Will return 32 random elements

'''
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,  # The input features to our model
    hidden_units=[400, 200],  # Two layers, each with 10 neurons
    n_classes=3,
    model_dir=PATH)

classifier.train(
    input_fn=lambda: my_input_fn(FILE_TRAIN, True, 8))

# Evaluate our model using the examples contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
evaluate_result = classifier.evaluate(
    input_fn=lambda: my_input_fn(FILE_TEST, False, 4))
print("Evaluation results")
for key in evaluate_result:
    print("   {}, was: {}".format(key, evaluate_result[key]))

# Predict the type of some Iris flowers.
# Let's predict the examples in FILE_TEST, repeat only once.
predict_results = classifier.predict(
    input_fn=lambda: my_input_fn(FILE_TEST, False, 1))
print("Predictions on test file")
for prediction in predict_results:
    # Will print the predicted class, i.e: 0, 1, or 2 if the prediction
    # is Iris Sentosa, Vericolor, Virginica, respectively.
    print(prediction["class_ids"][0])
'''
