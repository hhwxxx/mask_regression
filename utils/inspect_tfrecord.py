from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
import os
import shutil

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

import input_pipeline

SAVE_DIR = './inspect_tfrecord'
TFRECORD_DIR = '../tfrecords/quadrilateral_2'
DATASET_SPLIT = 'train'


def main(tfrecord_dir, dataset_split):
    data = input_pipeline.inputs(tfrecord_dir, dataset_split, True, 1, 1)
    image, label = data
    image = tf.squeeze(image)
    label = tf.squeeze(label)
    
    image = image.numpy()
    label = label.numpy()
    
    image *= 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    label *= 224
    label = label.astype(np.int32)
    
    label = np.split(label, 2, axis=0)
    for _label in label:
        _label = list(_label)
        _label = (_label[0:2][::-1] + _label[2:4][::-1]
                  + _label[-2:][::-1] + _label[4:6][::-1])
        draw.polygon(_label, outline=(255, 0, 0))
    
    image.save(os.path.join(SAVE_DIR, 'test.png'), 'PNG')


if __name__ == '__main__':
    tf.enable_eager_execution()
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)
    main(TFRECORD_DIR, DATASET_SPLIT)
