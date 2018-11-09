from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models

slim = tf.contrib.slim


MODEL_MAP = {
    'custom': models.custom,
    'vgg_16': models.vgg_16,
    'resnet_50': models.resnet_50,
}


def inference(model_variant, images, is_training=True):
    model = MODEL_MAP[model_variant]
    predictions = model(images, is_training)
    predictions = tf.squeeze(predictions)

    return predictions


def loss(predictions, labels):
    # absolute difference
    # absolute_loss = tf.losses.absolute_difference(
    #     labels=labels, predictions=predictions, scope='absolute_difference_loss')
    
    # mean squared error
    mean_squared_error = tf.losses.mean_squared_error(
        labels=labels, predictions=predictions, scope='mean_squared_error')

    # regularization loss
    regularization_loss = tf.losses.get_regularization_loss()

    total_loss = mean_squared_error + regularization_loss
    total_loss = tf.identity(total_loss, 'total_loss')

    tf.summary.scalar('losses/mean_square_loss', mean_squared_error)
    tf.summary.scalar('losses/regularization_loss', regularization_loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    return total_loss