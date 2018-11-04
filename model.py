from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import input_pipeline

slim = tf.contrib.slim

WEIGHT_DECAY = 0.0001
DROPOUT_KEEP_PROB = 0.5
BATCH_NORM_DECAY = 0.9


def inference(images, is_training=True):
    images.set_shape([None, input_pipeline.IMAGE_SHAPE[0], input_pipeline.IMAGE_SHAPE[1], 1])
    batch_norm_params = {
        'is_training': is_training,
        'decay': BATCH_NORM_DECAY,
        'epsilon': 1e-5,
        'scale': True,
        'fused': True,
    }

    with tf.variable_scope('mask_regression'):
        with slim.arg_scope([slim.model_variable, slim.variable],
                            device='/cpu:0'):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
                                # normalizer_fn=slim.batch_norm):
                # with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                net = slim.repeat(images, 2, slim.conv2d, 32, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = tf.reduce_mean(net, axis=[1, 2], keep_dims=True)
                net = slim.conv2d(net, 64, [1, 1], scope='conv5')
                net = slim.dropout(net, DROPOUT_KEEP_PROB, is_training=is_training, 
                        scope='dropout')

                predictions = slim.conv2d(net, 8, [1, 1], activation_fn=None, 
                        normalizer_fn=None, scope='conv6')
                predictions = tf.squeeze(predictions)

    return predictions


def loss(predictions, labels):
    # absolute difference
    absolute_loss = tf.losses.absolute_difference(
        labels=labels, predictions=predictions, scope='absolute_difference_loss'
    )
    
    # mean squared error
    mean_squared_error = tf.losses.mean_squared_error(
        labels=labels, predictions=predictions, scope='mean_squared_error'
    )

    # regularization loss
    regularization_loss = tf.losses.get_regularization_loss()

    total_loss = mean_squared_error
    # total_loss = absolute_loss
    total_loss = tf.identity(total_loss, 'total_loss')

    tf.summary.scalar('losses/mean_square_loss', mean_squared_error)
    # tf.summary.scalar('losses/regularization_loss', regularization_loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    return total_loss