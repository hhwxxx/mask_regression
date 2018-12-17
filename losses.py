from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf


class Loss(object):
    """Abstract base class for loss functions.
    """
    __metaclass__ = ABCMeta

    def __call__(self,
                 prediction_tensor,
                 target_tensor,
                 modify_zero_targets=True,
                 ignore_zero_targets=False,
                 scope=None):
        """Call the loss object.

        Args:
            prediction_tensor: A float tensor of shape [batch, num_points]
                representing the predicted values of points.
            target_tensor: A float tensor of shape [batch, num_points]
                representing the regression targets.
            modify_zero_targets: whether to modify zeros targets to avoid
                dividing by zero.
            ignore_zero_targets: whether to ignore nan targets in the loss computation.
            scope: Op scope name. Defaults to 'Loss' if None.

        Returns:
            loss: a tensor representing the value of the loss function.

        Raises:
            ValueError: if both modify_zero_targets and ignore_zero_targets
                        are True.
        """
        with tf.name_scope(
                scope, 'Loss', [prediction_tensor, target_tensor]) as scope:
            if modify_zero_targets and ignore_zero_targets:
                raise ValueError('modify_zero_targets and ignore_zero_targets '
                                 'should not be True at the same time.')
            if modify_zero_targets:
                target_tensor = tf.where(
                    tf.equal(target_tensor, 0),
                    target_tensor + 1e-3,
                    target_tensor)
            if ignore_zero_targets:
                target_tensor = tf.where(tf.equal(target_tensor, 0),
                                         prediction_tensor,
                                         target_tensor)
            return self._compute_loss(prediction_tensor, target_tensor)

    @abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor):
        """Method to be overridden by implementations.

        Args:
            prediction_target: A float tensor of shape [batch, num_points]
                representing the predicted values of points.
            target_tensor: A float tensor of shape [batch, num_points]
                representing the regression targets.

        Returns:
            loss: a float scalar tensor representing the loss value.
        """
        pass


class DistanceLoss(Loss):
    """Euclidean distance Loss.
    """
    def __init__(self, weights):
        """Constructor.

        Args:
            weights: a list of integer representing the weights.
        """
        self._weights = tf.constant(weights, dtype=tf.float32)

    def _compute_loss(self, prediction_tensor, target_tensor):
        """Compute mean squared error of distance between labels 
        and predictions.

        Args:
            prediction_tensor: model predictions with shape (batch, 16).
            target_tensor: ground truth with shape (batch, 16).

        Returns:
            distance_loss: mean squared error of distances.
        """
        prediction_tensor = tf.split(prediction_tensor, num_or_size_splits=2, axis=1)
        target_tensor = tf.split(target_tensor, num_or_size_splits=2, axis=1)

        distance_loss_list = []
        for prediction, label in zip(prediction_tensor, target_tensor):
            distance_loss_list.append(self._compute_distance_loss(prediction, label))

        distance_loss_list = tf.convert_to_tensor(distance_loss_list)
        distance_loss = tf.reduce_sum(distance_loss_list * self._weights)

        return distance_loss

    def _compute_distance_loss(self, predictions, labels):
        """Compute mean squared error of distances.

        Args:
            predictions: model predictions with shape (batch, 8).
            labels: ground truth with shape (batch, 8).

        Returns:
            distance_loss: mean squared error of distances.
        """
        # extract labels corner points
        # each point should have shape (batch, 2)
        labels_top_left, labels_top_right, labels_bottom_left, \
            labels_bottom_right = tf.split(
                labels, num_or_size_splits=4, axis=1)

        # extract predictions corner points
        # each point should have shape (batch, 2)
        predictions_top_left, predictions_top_right, predictions_bottom_left, \
            predictions_bottom_right = tf.split(
                predictions, num_or_size_splits=4, axis=1)

        # distance between labels corner points
        # each distance should have shape (batch, 1)
        labels_top_left_top_right = self._euclidean_distance(
            labels_top_left, labels_top_right)
        labels_top_right_bottom_right = self._euclidean_distance(
            labels_top_right, labels_bottom_right)
        labels_bottom_left_bottom_right = self._euclidean_distance(
            labels_bottom_left, labels_bottom_right)
        labels_top_left_bottom_left = self._euclidean_distance(
            labels_top_left, labels_bottom_left)

        # distance between predictions corner points
        # each distance should have shape (batch, 1)
        predictions_top_left_top_right = self._euclidean_distance(
            predictions_top_left, predictions_top_right)
        predictions_top_right_bottom_right = self._euclidean_distance(
            predictions_top_right, predictions_bottom_right)
        predictions_bottom_left_bottom_right = self._euclidean_distance(
            predictions_bottom_left, predictions_bottom_right)
        predictions_top_left_bottom_left = self._euclidean_distance(
            predictions_top_left, predictions_bottom_left)

        # labels_distance should have shape (batch, 4)
        labels_distance =  tf.concat([labels_top_left_top_right, 
                                      labels_top_right_bottom_right,
                                      labels_bottom_left_bottom_right, 
                                      labels_top_left_bottom_left], 
                                     axis=1)
        # predictions_distance should have shape (batch, 4)
        predictions_distance = tf.concat([predictions_top_left_top_right, 
                                          predictions_top_right_bottom_right,
                                          predictions_bottom_left_bottom_right,
                                          predictions_top_left_bottom_left], 
                                         axis=1)

        distance_loss = tf.losses.mean_squared_error(
            labels=labels_distance, predictions=predictions_distance, 
            scope='distance_loss')

        return distance_loss

    def _euclidean_distance(self, x, y):
        """Compute Euclidean distance between points x and y.

        Args:
            x: A point whose shape is (batch, 2). 
                Second dimension represents (height, width).
            y: A point whose shape is (batch, 2). 
                Second dimension represents (height, width).

        Returns:
            distance: Euclidean distance between x and y. 
                Its shape shoule be (batch, 1).
        """
        distance = tf.sqrt(
            tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

        return distance


class CosineLoss(Loss):
    """Cosine loss.
    """
    def __init__(self, weights):
        """Constructor.

        Args:
            weights: a list of integer representing the weights.
        """
        self._weights = tf.constant(weights, dtype=tf.float32)

    def _compute_loss(self, prediction_tensor, target_tensor):
        """Compute mean squared error of cosine.

        Args:
            prediction_tensor: model predictions with shape (batch, 8).
            target_tensor: ground truth with shape (batch, 8).

        Returns:
            cosine_loss: mean squared error of cosine.
        """
        prediction_tensor = tf.split(prediction_tensor, num_or_size_splits=2, axis=1)
        target_tensor = tf.split(target_tensor, num_or_size_splits=2, axis=1)

        cosine_loss_list = []
        for prediction, label in zip(prediction_tensor, target_tensor):
            #_individual_cosine_loss = tf.where(
            #    tf.equal(tf.reduce_sum(label), tf.constant(0, dtype=tf.float32)), 
            #    tf.constant(0, dtype=tf.float32), 
            #    _compute_cosine_loss(prediction, label)) 
            #cosine_loss_list.append(_individual_cosine_loss)
            cosine_loss_list.append(self._compute_cosine_loss(prediction, label))

        cosine_loss_list = tf.convert_to_tensor(cosine_loss_list)
        cosine_loss = tf.reduce_sum(cosine_loss_list * self._weights)

        return cosine_loss

    def _compute_cosine_loss(self, predictions, labels):
        # extract labels corner points
        # each point should have shape (batch, 2)
        labels_top_left, labels_top_right, labels_bottom_left, \
            labels_bottom_right = tf.split(labels, num_or_size_splits=4, axis=1)

        # extract predictions corner points
        # each point should have shape (batch, 2)
        predictions_top_left, predictions_top_right, predictions_bottom_left, \
            predictions_bottom_right = tf.split(
                predictions, num_or_size_splits=4, axis=1)

        # cosine between labels corner points
        # each should have shape (batch, 1) 
        labels_top_left_top_right = self._compute_cosine(
            labels_top_left, labels_top_right)
        labels_top_right_bottom_right = self._compute_cosine(
            labels_top_right, labels_bottom_right)
        labels_bottom_left_bottom_right = self._compute_cosine(
            labels_bottom_left, labels_bottom_right)
        labels_top_left_bottom_left = self._compute_cosine(
            labels_top_left, labels_bottom_left)

        # cosine between predictions corner points
        # each cosine should have shape (batch, 1) 
        predictions_top_left_top_right = self._compute_cosine(
            predictions_top_left, predictions_top_right)
        predictions_top_right_bottom_right = self._compute_cosine(
            predictions_top_right, predictions_bottom_right)
        predictions_bottom_left_bottom_right = self._compute_cosine(
            predictions_bottom_left, predictions_bottom_right)
        predictions_top_left_bottom_left = self._compute_cosine(
            predictions_top_left, predictions_bottom_left)

        # labels_cosine should have shape (batch, 4)
        labels_cosine =  tf.concat([labels_top_left_top_right, 
                                    labels_top_right_bottom_right,
                                    labels_bottom_left_bottom_right, 
                                    labels_top_left_bottom_left], 
                                   axis=1)
        # predictions_cosine should have shape (batch, 4)
        predictions_cosine = tf.concat([predictions_top_left_top_right, 
                                        predictions_top_right_bottom_right,
                                        predictions_bottom_left_bottom_right,
                                        predictions_top_left_bottom_left], 
                                       axis=1)

        # may contain outlier
        cosine_loss = tf.losses.mean_squared_error(
            labels=labels_cosine, predictions=predictions_cosine, 
            scope='cosine_loss')

        return cosine_loss

    def _compute_slope(self, x, y):
        """Compute slope between points x and y.

        slope = delta_height / delta_width

        Args:
            x: A point whose shape is (batch, 2). 
                Second dimension represents (height, width).
            y: A point whose shape is (batch, 2). 
                Second dimension represents (height, width).

        Returns:
            slope: slope between x and y. 
                Its shape shoule be (batch, 1).
        """
        epsilon = 1e-5  # avoid x divided by 0
        delta = y - x
        delta_height, delta_width = tf.split(
            delta, num_or_size_splits=2, axis=1)
        slope = delta_height / (delta_width + epsilon)

        return slope

    def _compute_cosine(self, x, y):
        """Compute cosine between points x and y.

        cosine = delta_width / distance(x, y)

        Args:
            x: A point whose shape is (batch, 2). 
                Second dimension represents (height, width).
            y: A point whose shape is (batch, 2). 
                Second dimension represents (height, width).

        Returns:
            cosine: cosine between x and y. 
                Its shape shoule be (batch, 1).
        """
        delta = y - x
        delta_height, delta_width = tf.split(
            delta, num_or_size_splits=2, axis=1)
        distance = self._euclidean_distance(x, y)

        epsilon = 1e-3
        cosine = delta_width / (distance + epsilon)
        # cosine = delta_width / distance

        return cosine

    def _euclidean_distance(self, x, y):
        """Compute Euclidean distance between points x and y.

        Args:
            x: A point whose shape is (batch, 2). 
                Second dimension represents (height, width).
            y: A point whose shape is (batch, 2). 
                Second dimension represents (height, width).

        Returns:
            distance: Euclidean distance between x and y.
                Its shape shoule be (batch, 1).
        """
        distance = tf.sqrt(
            tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

        return distance


class PointMSE(Loss):
    """Point mean squared error.
    """
    def __init__(self, weights):
        """Constructor.

        Args:
            weights: a list of integer representing the weights.
        """
        self._weights = tf.constant(weights, dtype=tf.float32)

    def _compute_loss(self, prediction_tensor, target_tensor):
        prediction_tensor = tf.split(prediction_tensor, num_or_size_splits=2, axis=1)
        target_tensor = tf.split(target_tensor, num_or_size_splits=2, axis=1)
        
        mse_list = []
        for prediction, label in zip(prediction_tensor, target_tensor):
            mse_list.append(
                self._compute_mse_loss(predictions=prediction, labels=label))

        mse_list = tf.convert_to_tensor(mse_list)
        mse_loss = tf.reduce_sum(mse_list * self._weights)

        return mse_loss

    def _compute_mse_loss(self, predictions, labels):
        return tf.losses.mean_squared_error(
            labels=labels, predictions=predictions, scope='mean_squared_error')
