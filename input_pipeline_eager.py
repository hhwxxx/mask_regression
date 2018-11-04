from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import functools
import os

tf.enable_eager_execution()

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_folder', './tfrecords', 'Folder containing tfrecords.')

IMAGE_SHAPE = (512, 512, 1)

NUMBER_TRAIN_DATA = 5000
NUMBER_VAL_DATA = 1500
NUMBER_TEST_DATA = 2000


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/data': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'image/width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'image/channel': tf.FixedLenFeature([], tf.int64, default_value=1),
            'image/name': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], tf.string, default_value=''),
            'label/top_left_height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'label/top_left_width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'label/top_right_height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'label/top_right_width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'label/bottom_left_height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'label/bottom_left_width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'label/bottom_right_height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'label/bottom_right_width': tf.FixedLenFeature([], tf.int64, default_value=0),
        }
    )

    image = tf.image.decode_jpeg(features['image/data'], channels=1)

    top_left_height = features['label/top_left_height'] / IMAGE_SHAPE[0]
    top_left_width = features['label/top_left_width'] / IMAGE_SHAPE[1]
    top_right_height = features['label/top_right_height'] / IMAGE_SHAPE[0]
    top_right_width = features['label/top_right_height'] / IMAGE_SHAPE[1]
    bottom_left_height = features['label/bottom_left_height'] / IMAGE_SHAPE[0]
    bottom_left_width = features['label/bottom_left_width'] / IMAGE_SHAPE[1]
    bottom_right_height = features['label/bottom_right_height'] / IMAGE_SHAPE[0]
    bottom_right_width = features['label/bottom_right_width'] / IMAGE_SHAPE[1]

    coordinates = tf.stack([top_left_height, top_left_width, top_right_height, top_right_width,
        bottom_left_height, bottom_left_width, bottom_right_height, bottom_right_width])

    return image, coordinates


def shift_image(image, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform([], 
                                                  -width_shift_range * IMAGE_SHAPE[1],
                                                  width_shift_range * IMAGE_SHAPE[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform([],
                                                   -height_shift_range * IMAGE_SHAPE[0],
                                                   height_shift_range * IMAGE_SHAPE[0])
        # Translate both 
        image = tf.contrib.image.translate(image, [width_shift_range, height_shift_range])

    return image


def flip_image(horizontal_flip, image):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        image = tf.cond(tf.less(flip_prob, 0.5), 
                                          lambda: tf.image.flip_left_right(image), 
                                          lambda: image)

    return image


def normalize(image):
    # normalize image to [0, 1]
    image = (1.0 / 255.0) * tf.to_float(image)

    return image


def augment(image,
            coordinates,
            resize=None,  # Resize the image to some size e.g. [512, 512]
            hue_delta=0,  # Adjust the hue of an RGB image by random factor
            horizontal_flip=False,  # Random left right flip,
            width_shift_range=0,  # Randomly translate the image horizontally
            height_shift_range=0):  # Randomly translate the image vertically 
    if resize is not None:
        # Resize both images
        image = tf.image.resize_images(image, resize, align_corners=True, 
                                       method=tf.image.ResizeMethod.BILINEAR)
    
    if hue_delta:
        image = tf.image.random_hue(image, hue_delta)
    
    image = flip_image(horizontal_flip, image)
    image = shift_image(image, width_shift_range, height_shift_range)
    image = normalize(image)
    
    return image, coordinates


train_config = {
    'resize': [IMAGE_SHAPE[0], IMAGE_SHAPE[1]],
    'hue_delta': 0,
    'horizontal_flip': False,
    'width_shift_range': 0,
    'height_shift_range': 0
}
train_preprocessing_fn = functools.partial(augment, **train_config)

val_config = {
    'resize': [IMAGE_SHAPE[0], IMAGE_SHAPE[1]],
}
val_preprocessing_fn = functools.partial(augment, **val_config)


test_config = {
    'resize': [IMAGE_SHAPE[0], IMAGE_SHAPE[1]],
}
config_preprocessing_fn = functools.partial(augment, **test_config)


def inputs(dataset_split, is_training, batch_size, num_epochs=None):
    filename = os.path.join(FLAGS.tfrecord_folder, dataset_split + '.tfrecord')

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)

        dataset = dataset.map(decode)

        if is_training:
            dataset = dataset.map(train_preprocessing_fn)
        else:
            dataset = dataset.map(val_preprocessing_fn)

        min_queue_examples = int(NUMBER_VAL_DATA * 0.3)
        if is_training:
            min_queue_examples = int(NUMBER_TRAIN_DATA * 0.3)
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=min_queue_examples)
    
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


if __name__ == '__main__':
    data = inputs('train', True, 32, None)
    print(data)