from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models

slim = tf.contrib.slim


MODEL_MAP = {
    'custom': models.custom,
    'vgg_16': models.vgg_16,
    'resnet_v2_50': models.resnet_v2_50,
    'resnet_v1_50_beta': models.resnet_v1_50_beta,
    'resnet_v1_101_beta': models.resnet_v1_101_beta,
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
