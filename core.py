from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models

import input_pipeline

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


def _bounding_box(points):
    """Compute bounding box of arbitrary quadrilateral.
    
    Args:
        points: corner points of quadrilateral with shape (batch_size, 8).
    Return:
        bounding_box: bounding box corner points with shape (batch_size, 4).
    """
    top_left, top_right, bottom_left, bottom_right = tf.split(
        points, num_or_size_splits=4, axis=1)
    # following variable should have shape (batch_size, )
    y_min = tf.minimum(top_left[:, 0], top_right[:, 0])
    y_max = tf.maximum(bottom_left[:, 0], bottom_right[:, 0])
    x_min = tf.minimum(top_left[:, 1], bottom_left[:, 1])
    x_max = tf.minimum(top_right[:, 1], bottom_right[:, 1])

    bounding_box = tf.stack(
        [y_min, x_min, y_max, x_max], axis=1)

    return bounding_box


def _is_inside_point(point, corner_points):
    """Check whether a point is inside the quadrilateral.
    Args:
        point -- a list representing point with length 2. [y, x]
        corner_points -- corner points of quadrilateral with shape (8, )
    """
    top_left, top_right, bottom_left, bottom_right = tf.split(
        corner_points, num_or_size_splits=4, axis=0)
    
    def _cross_product(P, Q):
        """Compute cross product of P and Q.
        Args:
            P -- point [h, w]
            Q -- point [h, w]
        """
        return P[1] * Q[0] - P[0] * Q[1] 

    # compute cross product x1y2 - x2y1
    # AP * AB
    # a = (point[1] - top_left[1]) * (top_right[0] - top_left[0]) - \
    #     (top_right[1] - top_left[1]) * (point[0] - top_left[0])
    a = _cross_product(point - top_left, top_right - top_left)
    b = _cross_product(point - top_right, bottom_right - top_right)
    c = _cross_product(point - bottom_right, bottom_left - bottom_right)
    d = _cross_product(point - bottom_left, top_left - bottom_left)

    if (a >= 0 and b >= 0 and c >= 0 and d >= 0) or \
        (a <= 0 and b <= 0 and c <= 0 and d <= 0):
        return True
    else:
        return False


def _compute_mask(bounding_box, corner_points):
    """Compute mask covering quadrilateral content.
    Args:
        bounding_box -- bouding box of quadrilateral with shape (4, )
        corner_points -- corner_points with shape (8, )
    """
    # mask = tf.zeros(shape=input_pipeline.IMAGE_SHAPE[:2])
    mask = tf.zeros(input_pipeline.IMAGE_SHAPE[:2])
    y_min, x_min, y_max, x_max = tf.split(
        bounding_box, num_or_size_splits=4, axis=0)
        
    # bbox_top_left = [y_min, x_min]
    # bbox_top_right = [y_min, x_max]
    # bbox_bottom_left = [y_max, x_min]
    # bbox_bottom_right = [y_max, x_max]

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if _is_inside_point([x, y], corner_points):
                # TODO (can not work)
                mask[y][x] = True
    
    return mask


def iou_loss(predictions, labels):
    """Compute IoU loss of two quadirlaterals.

    Args:
        predictions -- predictions with shape (batch_size, 8).
        labels -- ground truth with shape (batch_size, 8).
    Return:
        iou_loss -- IoU Loss.
    """
    batch_size = tf.shape(predictions)[0]

    predictions_bbox = _bounding_box(predictions)
    labels_bbox = _bounding_box(labels)

    iou_loss_list = []
    for i in range(batch_size):
        prediction_mask = _compute_mask(predictions_bbox[i], predictions[i])
        label_mask = _compute_mask(labels_bbox[i], labels[i])

        # TODO

    iou_loss = tf.reduce_mean(iou_loss_list)

    return iou_loss
    

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
        predictions -- predictions with shape (batch_size, 8).
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


def distance_loss(predictions, labels):
    """Compute distance loss.

    Args:
        predictions: float32 tensor of shape (batch_size, 16).
        labels: float32 tensor of shape (batch_size, 16).
    """
    predictions = tf.split(predictions, num_or_size_splits=2, axis=1)
    labels = tf.split(labels, num_or_size_splits=2, axis=1)

    distance_loss_list = []
    for prediction, label in zip(predictions, labels):
        distance_loss_list.append(_distance_loss(prediction, label))
    
    distance_loss_ = tf.reduce_mean(distance_loss_list)
    tf.identity(distance_loss_, 'distance_loss')

    return distance_loss_
        

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
        predictions -- predictions with shape (batch_size, 16).
        labels -- ground truth with shape (batch_size, 2, 8)
    Return:
        total_loss -- loss.
    """

    # Reshape labels to have shape (batch_size, 16).
    labels = tf.reshape(labels, [-1, models.NUMBER_OUTPUT])

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
    distance_loss_ = distance_loss(predictions, labels)

    # cosine loss
    cosine_loss = _cosine_loss(predictions, labels)

    # regularization loss
    regularization_loss = tf.losses.get_regularization_loss()

    # without regularization loss
    # total_loss = mean_squared_error + distance_loss + cosine_loss
    # without cosine_loss
    total_loss = mean_squared_error + distance_loss_
    total_loss = tf.identity(total_loss, 'total_loss')

    tf.summary.scalar('losses/mean_square_loss', mean_squared_error)
    tf.summary.scalar('losses/distane_loss', distance_loss_)
    tf.summary.scalar('losses/cosine_loss', cosine_loss)
    tf.summary.scalar('losses/regularization_loss', regularization_loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    return total_loss