# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be here
def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""
    # Input Layer
    # black and white picture
    """
      filter=5 ,the filter's num
      kernel_size , filter's width*heigth
    """
    input_layer = tf.reshape(features['x'],[-1,28,28, 1])
    conv1 = tf.layers.conv2d(
            inputs=input_layer, 
            filters=32, 
            kernel_size=[5,5],
            padding="same", 
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu) 
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1,7*7*64])
    
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    
    # Our output tensor dropout has shape [batch_size, 1024]
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Our final output densor of CNN, logits , has shape[bathc_size, 1024] 
    logits = tf.layers.dense(inputs=dropout, units=10)
    
    #tf.argmax(input=logits, axis=1)

    #tf.nn.softmaxlogits, name="softmax_tensor"()

    predictions = {
        "class":tf.argmax(input=logits, axis=1),
        "probabilities":tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),depth=10)

    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32), depth=10)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
            "accuracy":tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loess, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # create the Estimator

    mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir="/tmp/mnist_convent_model")
    
    # set up a logging Hook

    tensors_to_log = {"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
     
    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)

    mnist_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])
    
    # Evaluate the model and print result

    eval_input_tf = tf.estimator.inputs.numpy_input_fn(
            x={"x":eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
    eval_result = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_result)


if __name__ == '__main__':
    tf.app.run()
