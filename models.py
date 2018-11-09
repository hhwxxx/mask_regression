from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from slim.nets import resnet_v2
# from tensorflow.contrib.slim.nets import resnet_v2


slim = tf.contrib.slim

WEIGHT_DECAY = 0.0001
DROPOUT_KEEP_PROB = 0.5
BATCH_NORM_DECAY = 0.9
NUMBER_OUTPUT = 8


exclude_list_custom = []
exclude_list_vgg_16 = ['vgg_16/predictions']
exclude_list_resnet_50 = ['resnet_v2_50/logits']

EXCLUDE_LIST_MAP = {
    'custom': exclude_list_custom,
    'vgg_16': exclude_list_vgg_16,
    'resnet_50': exclude_list_resnet_50,
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


def resnet_50(images, is_training):
    # images size is (None, 224, 224, 3), which is equal to default image size of ResNet-50.
    # net is final output without activation.
    with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=BATCH_NORM_DECAY)):
        net, end_points = resnet_v2.resnet_v2_50(
            inputs=images, num_classes=8, is_training=True, global_pool=True,
            spatial_squeeze=True, scope='resnet_v2_50')
    
    return net
