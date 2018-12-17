from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models

import losses

slim = tf.contrib.slim

MODEL_MAP = {
    'custom': models.custom,
    'vgg_16': models.vgg_16,
    'resnet_v2_50': models.resnet_v2_50,
    'resnet_v1_50_beta': models.resnet_v1_50_beta,
    'resnet_v1_101_beta': models.resnet_v1_101_beta,
    'resnet_v1_50_beta_lstm': models.resnet_v1_50_beta_lstm,
}


def inference(model_variant, images, is_training=True):
    """Compute predictions of the network.

    Args:
        model_variant -- String. Specify network backbone.
        images -- Input images with shape (batch_size, height, width, 3).
        is_training -- Training mode or inference mode. 
                       Relevant to BN layers and dropout layers.
    
    Return:
        predictions -- Model output with shape (batch_size, 8).
    """
    model = MODEL_MAP[model_variant]
    predictions = model(images, is_training)
    predictions = tf.squeeze(predictions)

    return predictions


def loss(predictions, labels, weights):
    """Compute total loss.

    Args:
        predictions: a float tensor of shape (batch, 16) representing 
            predicted value.
        labels: a float tensor with shape (batch, 16) representing
            ground truth.
        weights: a list of integer representing weights of two output parts.

    Returns:
        total_loss: a float tensor representing value of total loss.
    """
    # point mean squared error
    point_mse = losses.PointMSE(weights)(
        predictions, labels, ignore_zero_targets=False, scope='PointMSE')

    # distance loss
    distance_loss = losses.DistanceLoss(weights)(
        predictions, labels, ignore_zero_targets=False, scope='DistanceLoss')

    # cosine_loss
    cosine_loss = losses.CosineLoss(weights)(
        predictions, labels, ignore_zero_targets=False, scope='CosineLoss')

    # regularization loss
    regularization_loss = tf.losses.get_regularization_loss()

    # total_loss without regularization_loss
    total_loss = 3 * point_mse + distance_loss + 2 * cosine_loss
    # total_loss = 3 * mean_squared_error + distance_loss
    total_loss = tf.identity(total_loss, 'total_loss')

    tf.summary.scalar('losses/point_loss', point_mse)
    tf.summary.scalar('losses/distance_loss', distance_loss)
    tf.summary.scalar('losses/cosine_loss', cosine_loss)
    tf.summary.scalar('losses/regularization_loss', regularization_loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    return total_loss
