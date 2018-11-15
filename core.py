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
    """Compute predictions of the network.

    Args:
        model_variant -- String. Specify network backbone.
        images -- Input images with shape (batch_size, height, width, 3)
        is_training -- Training mode or inference mode.
                       Revelant to BN layers and dropout layers.
    
    Return:
        predictions -- Network output with shape (batch_size, 8)
    """
    model = MODEL_MAP[model_variant]
    predictions = model(images, is_training)
    predictions = tf.squeeze(predictions)

    return predictions


def _euclidean_distance(x, y):
    """Compute Euclidean distance between points x and y.

    Args:
        x -- A point whose shape is (batch_size, 2).
             Second dimension represents (height, width).
        y -- A point whose shape is (batch_size, 2).
             Second dimension represents (height, width).
    
    Return:
        distance -- Euclidean distance between x and y.
                    Its shape should be (batch_size, 1).
    """
    distance = tf.sqrt(
        tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))
    
    return distance


def _distance_loss(predictions, labels):
    """Compute mean squared error of distances.

    Args:
        predictions -- model predictions with shape (batch_size, 8).
        labels -- ground truth with shape (batch_size, 8).
    Return:
        distance_loss -- mean squared error of distances.
    """
    # extract labels corner points
    # each point should have shape (batch_size, 2)
    labels_top_left, labels_top_right, labels_bottom_left, \
        labels_bottom_right = tf.split(labels, num_or_size_splits=4, axis=1)
    
    # extract predictions corner points
    # each point should have shape (batch_size, 2)
    predictions_top_left, predictions_top_right, predictions_bottom_left, \
        predictions_bottom_right = tf.split(
            predictions, num_or_size_splits=4, axis=1)
    
    # distance between labels corner points
    # each distance should have shape (batch_size, 1)
    labels_top_left_top_right = _euclidean_distance(
        labels_top_left, labels_top_right)
    labels_top_right_bottom_right = _euclidean_distance(
        labels_top_right, labels_bottom_right)
    labels_bottom_left_bottom_right = _euclidean_distance(
        labels_bottom_left, labels_bottom_right)
    labels_top_left_bottom_left = _euclidean_distance(
        labels_top_left, labels_bottom_left)
    
    # distance between predictions corner points
    # each distance should have shape (batch_size, 1)
    predictions_top_left_top_right = _euclidean_distance(
        predictions_top_left, predictions_top_right)
    predictions_top_right_bottom_right = _euclidean_distance(
        predictions_top_right, predictions_bottom_right)
    predictions_bottom_left_bottom_right = _euclidean_distance(
        predictions_bottom_left, predictions_bottom_right)
    predictions_top_left_bottom_left = _euclidean_distance(
        predictions_top_left, predictions_bottom_left)
    
    # labels_distance should have shape (batch_size, 4)
    labels_distance = tf.concat([labels_top_left_top_right,
                                 labels_top_right_bottom_right,
                                 labels_bottom_left_bottom_right,
                                 labels_top_left_bottom_left],
                                axis=1)
    # predictions_distance should have shape (batch_size, 4)
    predictions_distance = tf.concat([predictions_top_left_top_right,
                                      predictions_top_right_bottom_right,
                                      predictions_bottom_left_bottom_right,
                                      predictions_top_left_bottom_left],
                                     axis=1)
    
    distance_loss = tf.losses.mean_squared_error(
        labels=labels_distance, predictions=predictions_distance,
        scope='distance_loss')
    
    return distance_loss


def _cosine_loss(predictions, labels):
    """Compute mean squared error of cosine.

    Args:
        predictions -- predictions with shape (batch_size, 8).
        labels -- ground truth with shape (batch_size, 8).
    Return:
        cosine_loss -- mean squared error of cosine.
    """

    def _compute_slope(x, y):
        """Compute slope between points x and y.
        slope = delta_height / delta_width

        Args:
            x -- A point with shape (batch_size, 2).
                 Second dimension represents (height, width).
            y -- A point with shape (batch_szie, 2).
                 Second dimension represents (height, width).
        Return:
            slope -- slope between x and y.
                     Its shape should be (batch_size, 1).
        """
        epsilon = 1e-5  # avoid divide by 0
        delta = y - x
        delta_height, delta_width = tf.split(
            delta, num_or_size_splits=2, axis=1)
        slope = delta_height / (delta_width + epsilon)

        return slope
    

    def _compute_cosine(x, y):
        """Compute cosine between points x and y.
        cosine = delta_width / distance(x, y)

        Args:
            x -- A point with shape (batch_size, 2).
                 Second dimension represents (height, width).
            y -- A point with shape (batch_szie, 2).
                 Second dimension represents (height, width).
        Return:
            cosine -- cosine between x and y.
                      Its shape should be (batch_size, 1).
        """
        delta = y - x
        delta_height, delta_width = tf.split(
            delta, num_or_size_splits=2, axis=1)
        distance = _euclidean_distance(x, y)

        cosine = delta_width / distance

        return cosine
    
    # extract labels corner points
    # each point should have shape (batch_size, 2)
    labels_top_left, labels_top_right, labels_bottom_left, \
        labels_bottom_right = tf.split(labels, num_or_size_splits=4, axis=1)
    
    # extract predictions corner points
    # each point should have shape (batch_size, 2)
    predictions_top_left, predictions_top_right, predictions_bottom_left, \
        predictions_bottom_right = tf.split(
            predictions, num_or_size_splits=4, axis=1)
    
    # corner_points = [(labels_top_left, labels_top_right),
    #                  (labels_top_right, labels_bottom_right),
    #                  (labels_bottom_left, labels_bottom_right),
    #                  (labels_top_left, labels_bottom_left)]
    # cosine between labels corner points
    # each should have shape (batch_size, 1)
    labels_top_left_top_right = _compute_cosine(
        labels_top_left, labels_top_right)
    labels_top_right_bottom_right = _compute_cosine(
        labels_top_right, labels_bottom_right)
    labels_bottom_left_bottom_right = _compute_cosine(
        labels_bottom_left, labels_bottom_right)
    labels_top_left_bottom_left = _compute_cosine(
        labels_top_left, labels_bottom_left)
    
    # cosine between predictions corner points
    # each cosine should have shape (batch_size, 1)
    predictions_top_left_top_right = _compute_cosine(
        predictions_top_left, predictions_top_right)
    predictions_top_right_bottom_right = _compute_cosine(
        predictions_top_right, predictions_bottom_right)
    predictions_bottom_left_bottom_right = _compute_cosine(
        predictions_bottom_left, predictions_bottom_right)
    predictions_top_left_bottom_left = _compute_cosine(
        predictions_top_left, predictions_bottom_left)
    
    # labels_cosine should have shape (batch_size, 4)
    labels_cosine = tf.concat([labels_top_left_top_right,
                               labels_top_right_bottom_right,
                               labels_bottom_left_bottom_right,
                               labels_top_left_bottom_left],
                              axis=1)
    # predictions_cosine should have shape (batch_size, 4)
    predictions_cosine = tf.concat([predictions_top_left_top_right,
                                    predictions_top_right_bottom_right,
                                    predictions_bottom_left_bottom_right,
                                    predictions_top_left_bottom_left],
                                   axis=1)
    
    cosine_loss = tf.losses.mean_squared_error(
        labels=labels_cosine, predictions=predictions_cosine,
        scope='cosine_loss')
    
    return cosine_loss


def loss(predictions, labels):
    """Compute loss.

    Args:
        predictions -- predictions with shape (batch_size, 8).
        labels -- ground truth with shape (batch_size, 8)
    Return:
        total_loss -- loss.
    """
    # absolute difference
    # absolute_loss = tf.losses.absolute_difference(
    #     labels=labels, predictions=predictions, scope='absolute_difference_loss')

    # huber loss
    # huber_loss = tf.losses.huber_loss(
    #     labels=labels, predictions=predictions, delta=1.35, scope='huber_loss')
    
    # mean squared error
    mean_squared_error = tf.losses.mean_squared_error(
        labels=labels, predictions=predictions, scope='mean_squared_error')
    
    # distance loss
    distance_loss = _distance_loss(predictions, labels)

    # cosine loss
    cosine_loss = _cosine_loss(predictions, labels)

    # regularization loss
    regularization_loss = tf.losses.get_regularization_loss()

    total_loss = mean_squared_error + distance_loss + cosine_loss + regularization_loss
    total_loss = tf.identity(total_loss, 'total_loss')

    tf.summary.scalar('losses/mean_square_loss', mean_squared_error)
    tf.summary.scalar('losses/distane_loss', distance_loss)
    tf.summary.scalar('losses/cosine_loss', cosine_loss)
    tf.summary.scalar('losses/regularization_loss', regularization_loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    return total_loss