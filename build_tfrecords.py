from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import glob
import six
import collections
import ast
from itertools import combinations

import tensorflow as tf
import pandas as pd

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('csv_dir', './mask_data/quadrilateral_multiple/', 
                    'Directory containing lists for training and validation.')
flags.DEFINE_string('output_dir', './tfrecords/quadrilateral_multiple/', 
                    'Directory to save tfrecord.')
flags.DEFINE_string('image_format', 'png', 'Image format')


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, image_format='jpeg', channels=3):
        """Class constructor.
        Args:
            image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
            channels: Image channels.
        """
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if self._image_format in ('jpeg', 'jpg', 'JPG'):
                self._decode = tf.image.decode_jpeg(self._decode_data,
                                                    channels=channels)
            elif self._image_format == 'png':
                self._decode = tf.image.decode_png(self._decode_data,
                                                   channels=channels)


    def read_image_dims(self, image_data):
        """Reads the image dimensions.
        Args:
            image_data: string of image data.
        Returns:
            image_height and image_width.
        """
        image = self.decode_image(image_data)

        return image.shape[:2]


    def decode_image(self, image_data):
        """Decodes the image data string.
        Args:
            image_data: string of image data.
        Returns:
            Decoded image data.
        Raises:
            ValueError: Value of image channels not supported.
        """
        image = self._session.run(self._decode,
                                  feed_dict={self._decode_data: image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')

        return image

def _int64_list_feature(values):
    """Returns a TF-Feature of int64_list.

    Args:
        values: A scalar or list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_list_feature(values):
    """Returns a TF-Feature of float_list.
    
    Args:
        values: A float or list of floats.
    
    Returns:
        A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
      values: A string.

  Returns:
      A TF-Feature.
  """
  def norm2bytes(value):
      return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def build_tfrecord(dataset_split):
    print('Processing {} data.'.format(dataset_split))

    data_csv = pd.read_csv(os.path.join(FLAGS.csv_dir, dataset_split + '.csv'))

    image_reader = ImageReader(image_format=FLAGS.image_format, channels=3)

    output_filename = os.path.join(FLAGS.output_dir, dataset_split + '.tfrecord')
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(len(data_csv)):
            image_filename = data_csv.iloc[i]['filename']
            image_name = image_filename.split('/')[-1]
            image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
            image_height, image_width = image_reader.read_image_dims(
                image_data)

            top_left_height = ast.literal_eval(
                str(data_csv.iloc[i]['top_left_height']))
            top_left_width = ast.literal_eval(
                str(data_csv.iloc[i]['top_left_width']))
            top_right_height = ast.literal_eval(
                str(data_csv.iloc[i]['top_right_height']))
            top_right_width = ast.literal_eval(
                str(data_csv.iloc[i]['top_right_width']))
            bottom_left_height = ast.literal_eval(
                str(data_csv.iloc[i]['bottom_left_height']))
            bottom_left_width = ast.literal_eval(
                str(data_csv.iloc[i]['bottom_left_width']))
            bottom_right_height = ast.literal_eval(
                str(data_csv.iloc[i]['bottom_right_height']))
            bottom_right_width = ast.literal_eval(
                str(data_csv.iloc[i]['bottom_right_width']))

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/data': _bytes_list_feature(image_data),
                        'image/height': _int64_list_feature(image_height),
                        'image/width': _int64_list_feature(image_width),
                        'image/channel': _int64_list_feature(3),
                        'image/name': _bytes_list_feature(image_name),
                        'image/format': _bytes_list_feature(FLAGS.image_format),
                        'label/top_left_height': _int64_list_feature(top_left_height),
                        'label/top_left_width': _int64_list_feature(top_left_width),
                        'label/top_right_height': _int64_list_feature(top_right_height),
                        'label/top_right_width': _int64_list_feature(top_right_width),
                        'label/bottom_left_height': _int64_list_feature(bottom_left_height),
                        'label/bottom_left_width': _int64_list_feature(bottom_left_width),
                        'label/bottom_right_height': _int64_list_feature(bottom_right_height),
                        'label/bottom_right_width': _int64_list_feature(bottom_right_width),
                    }
                )
            )
            tfrecord_writer.write(example.SerializeToString())
    print('Finished processing {} data.'.format(dataset_split))
  

def main(unused_argv):
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    
    dataset = ['train', 'val', 'test']
    for dataset_split in dataset:
        build_tfrecord(dataset_split)
        print('Finished processing', dataset_split)
    
    print('Finished.')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
