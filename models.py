from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_utils

from slim.nets import resnet_v2
import resnet_v1_beta

slim = tf.contrib.slim

WEIGHT_DECAY = 0.0001
DROPOUT_KEEP_PROB = 0.5
BATCH_NORM_DECAY = 0.9
NUMBER_OUTPUT = 16


exclude_list_custom = []
exclude_list_vgg_16 = ['vgg_16/predictions']
exclude_list_resnet_v2_50 = ['resnet_v2_50/logits']
exclude_list_resnet_v1_50_beta = ['resnet_v1_50/logits']
exclude_list_resnet_v1_101_beta = ['resnet_v1_101/logits']
exclude_list_resnet_v1_50_beta_lstm = ['lstm', 'logits']

EXCLUDE_LIST_MAP = {
    'custom': exclude_list_custom,
    'vgg_16': exclude_list_vgg_16,
    'resnet_v2_50': exclude_list_resnet_v2_50,
    'resnet_v1_50_beta': exclude_list_resnet_v1_50_beta,
    'resnet_v1_101_beta': exclude_list_resnet_v1_101_beta,
    'resnet_v1_50_beta_lstm': exclude_list_resnet_v1_50_beta_lstm,
}


def custom(images, is_training):
    """Custom model.
    VGG-like model.
    """
    batch_norm_params = {
        'is_training': is_training,
        'decay': BATCH_NORM_DECAY,
        'epsilon': 1e-5,
        'scale': True,
        'fused': True,
    }

    with tf.variable_scope('custom'):
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
                net = tf.reduce_mean(net, axis=[1, 2], keepdims=True)
                net = slim.conv2d(net, 64, [1, 1], scope='conv5')
                net = slim.dropout(net, DROPOUT_KEEP_PROB, is_training=is_training, 
                        scope='dropout')

                predictions = slim.conv2d(net, NUMBER_OUTPUT, [1, 1], 
                    activation_fn=None, normalizer_fn=None, scope='conv6')

    return predictions


def vgg_16(images, is_training):
    """VGG-16 based model.
    Input images size should be [batch_size, 224, 224, 3] in order to 
    initialize from ImageNet pretrained model.
    """

    batch_norm_params = {
        'is_training': is_training,
        'decay': BATCH_NORM_DECAY,
        'epsilon': 1e-5,
        'scale': True,
        'fused': True,
    }

    with tf.variable_scope('vgg_16'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
            # with slim.arg_scope([slim.batch_norm], **batch_norm_params) as scope:
            net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool5')

            # fully connected layer -> conv layer
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            net = slim.dropout(net, DROPOUT_KEEP_PROB, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, DROPOUT_KEEP_PROB, is_training=is_training, scope='dropout7')

            predictions = slim.conv2d(net, NUMBER_OUTPUT, [1, 1], activation_fn=None, 
                                      normalizer_fn=None, scope='predictions')
    
    return predictions


def resnet_v2_50(images, is_training):
    # images size is (None, 224, 224, 3), which is equal to default image size of ResNet-50.
    # net is final output without activation.
    fine_tune_batch_norm = False
    with slim.arg_scope(
            resnet_v2.resnet_arg_scope(batch_norm_decay=BATCH_NORM_DECAY)):
        net, end_points = resnet_v2.resnet_v2_50(
            inputs=images,
            num_classes=NUMBER_OUTPUT, 
            is_training=(is_training and fine_tune_batch_norm), 
            global_pool=True, 
            spatial_squeeze=True, 
            scope='resnet_v2_50')
    
    return net


def resnet_v1_50_beta(images, is_training):
    """ResNet-50 v1 beta.
    Replace first 7*7 conv layers with three 3*3 conv layers.
    """
    fine_tune_batch_norm = False
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        feature, end_points = resnet_v1_beta.resnet_v1_50_beta(
            images,
            num_classes=NUMBER_OUTPUT,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=True)
    
    return feature


def resnet_v1_101_beta(images, is_training):
    """ResNet-101 v1 beta.
    Replace first 7*7 conv layers with three 3*3 conv layers.
    """
    fine_tune_batch_norm = False
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        feature, end_points = resnet_v1_beta.resnet_v1_101_beta(
            images,
            num_classes=NUMBER_OUTPUT,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=True)
    
    return feature


def resnet_v1_50_beta_lstm(images, is_training):
    """ResNet-50 v1 beta with lstm head.
    Replace first 7*7 conv layers with three 3*3 conv layers.
    """
    fine_tune_batch_norm = False
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        feature, end_points = resnet_v1_beta.resnet_v1_50_beta(
            images,
            num_classes=None,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=True)

    with tf.variable_scope('lstm'):
        # feature has shape (batch_size, feature_size)
        feature = tf.squeeze(feature)
        # feature has shape (1, batch_size, feature_size)
        feature = tf.expand_dims(feature, axis=0)
        # feature has shape (time_step, batch_size, feature_size)
        feature = tf.tile(feature, (2, 1, 1))

        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=128)
        # lstm_output has shape (time_step, batch_size, 128)
        lstm_output, state = lstm(feature)
        # each element in lstm_output_list has shape (batch_size, 128)
        lstm_output_list = tf.unstack(lstm_output, axis=0)
    
    output_list = []
    with tf.variable_scope('logits'):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
            for element in lstm_output_list:
                feature = slim.fully_connected(element, NUMBER_OUTPUT // 2,
                                               activation_fn=None,
                                               normalizer_fn=None)
                output_list.append(feature)
            # output has shape (batch_size, NUMBER_OUTPUT)
            output = tf.concat(output_list, axis=1)

    return output
