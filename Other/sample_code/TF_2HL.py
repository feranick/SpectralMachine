import time
import tensorflow as tf

# We simulate some raw input data 
# (think about it as fetching some data from the file system)
# let's say: batches of 128 samples, each containing 1024 data points
x_inputs_data = tf.random_normal([128, 1024], mean=0, stddev=1)
# We will try to predict this law:
# predict 1 if the sum of the elements is positive and 0 otherwise
y_inputs_data = tf.cast(tf.reduce_sum(x_inputs_data, axis=1, keep_dims=True) > 0, tf.int32)

# We build our small model: a basic two layers neural net with ReLU
with tf.variable_scope("placeholder"):
    input = tf.placeholder(tf.float32, shape=[None, 1024])
    y_true = tf.placeholder(tf.int32, shape=[None, 1])
with tf.variable_scope('FullyConnected'):
    w = tf.get_variable('w', shape=[1024, 1024], 
initializer=tf.random_normal_initializer(stddev=1e-1))
    b = tf.get_variable('b', shape=[1024], initializer=tf.constant_initializer(0.1))
    z = tf.matmul(input, w) + b
    y = tf.nn.relu(z)

    w2 = tf.get_variable('w2', shape=[1024, 1], 
initializer=tf.random_normal_initializer(stddev=1e-1))
    b2 = tf.get_variable('b2', shape=[1], initializer=tf.constant_initializer(0.1))
    z = tf.matmul(y, w2) + b2
with tf.variable_scope('Loss'):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(y_true, tf.float32), z)
    loss_op = tf.reduce_mean(losses)
with tf.variable_scope('Accuracy'):
    y_pred = tf.cast(z > 0, tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")

# We add the training operation, ...
adam = tf.train.AdamOptimizer(1e-2)
train_op = adam.minimize(loss_op, name="train_op")

startTime = time.time()
with tf.Session() as sess:
    # ... init our variables, ...
    sess.run(tf.global_variables_initializer())

    # ... check the accuracy before training, ...
    x_input, y_input = sess.run([x_inputs_data, y_inputs_data])
    sess.run(accuracy, feed_dict={
        input: x_input,
        y_true: y_input
    })

    # ... train ...
    for i in range(5000):
        #  ... by sampling some input data (fetching) ...
        x_input, y_input = sess.run([x_inputs_data, y_inputs_data])
        # ... and feeding it to our model
        _, loss = sess.run([train_op, loss_op], feed_dict={
            input: x_input,
            y_true: y_input
        })

        # We regularly check the loss
        if i % 500 == 0:
            print('iter:%d - loss:%f' % (i, loss))

    # Finally, we check our final accuracy
    x_input, y_input = sess.run([x_inputs_data, y_inputs_data])
    sess.run(accuracy, feed_dict={
        input: x_input,
        y_true: y_input
    })

print("Time taken: %f" % (time.time() - startTime))
