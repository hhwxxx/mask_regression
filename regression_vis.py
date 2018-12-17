from __future__ import absolute_import
from __future__ import division
from __future__ import division

import os
import sys
import shutil
import sys

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

import input_pipeline
import core


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_dir',
                    './tfrecords/quadrilateral_multiple',
                    'Directory containing tfrecords.')
flags.DEFINE_string('dataset_split', 'test',
                    'Using which dataset split to train the network.')
flags.DEFINE_string('model_variant', 'vgg_16', 'Model to use.')
flags.DEFINE_string('checkpoint_dir',
                    './exp/vgg_quadrilateral_multiple_01/train',
                    'Directory containing trained checkpoints.')
flags.DEFINE_string('vis_dir',
                    './exp/vgg_quadrilateral_multiple_01/vis',
                    'Training directory.')

flags.DEFINE_integer('batch_size', 1, 'Batch size used for visualization.')
flags.DEFINE_boolean('is_training', False, 'Is training?')


def vis(model_variant, tfrecord_dir, dataset_split):
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            images, labels = input_pipeline.inputs(
                tfrecord_dir, dataset_split, FLAGS.is_training,
                FLAGS.batch_size, num_epochs=1)
        predictions = core.inference(model_variant, images,
                                     is_training=FLAGS.is_training)

        if not dataset_split in ['train', 'val', 'test']:
            raise Exception('Invalid argument.')
        elif dataset_split == 'train':
            num_iters = input_pipeline.NUMBER_TRAIN_DATA
        elif dataset_split == 'val':
            num_iters = input_pipeline.NUMBER_VAL_DATA
        elif dataset_split == 'test':
            num_iters = input_pipeline.NUMBER_TEST_DATA

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = int(
                ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print('Get global_step from checkpoint name')
        else:
            global_step = tf.train.get_or_create_global_step()
            print('Create global_step.')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                config=config,
                checkpoint_dir=FLAGS.checkpoint_dir)) as mon_sess:
            cur_iter = 0
            while cur_iter < num_iters:
                print('Visualizing {:06d}.png'.format(cur_iter))
                image, labels_, predictions_ = mon_sess.run(
                    [images, labels, predictions])
                # image shape is (224, 224, 3)
                image = np.squeeze(image)
                image = np.uint8(image * 255.0)
                labels_ = np.squeeze(labels_)
                
                predictions_ = predictions_ * input_pipeline.IMAGE_SHAPE[0]
                labels_ = labels_ * input_pipeline.IMAGE_SHAPE[0]
                
                predictions_ = list(np.split(predictions_, indices_or_sections=2))
                labels_ = list(np.split(labels_, indices_or_sections=2))
                pil_image = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_image)
                for prediction, label in zip(predictions_, labels_):
                    prediction = list(prediction)
                    label = list(label)
                    # top_left -> top_right -> bottom_right -> bottom_left
                    prediction = prediction[:4] + prediction[-2:] + prediction[4:6]
                    label = label[:4] + label[-2:] + label[4:6]
                    # point is represented as (width, height) in PIL
                    prediction = (prediction[0:2][::-1]
                                  + prediction[2:4][::-1]
                                  + prediction[4:6][::-1]
                                  + prediction[-2:][::-1])
                    label = (label[0:2][::-1] + label[2:4][::-1]
                             + label[4:6][::-1] + label[-2:][::-1])
                    
                    draw.polygon(label, outline=(0, 255, 0))
                    draw.polygon(prediction, outline=(255, 0, 0))

                pil_image.save('{}/{:06}.png'.format(FLAGS.vis_dir, cur_iter),
                               format='PNG')

                cur_iter += 1

            print('Finished!')


def main(unused_argv):
    if os.path.exists(FLAGS.vis_dir):
        shutil.rmtree(FLAGS.vis_dir)
    if not os.path.exists(FLAGS.vis_dir):
        os.makedirs(FLAGS.vis_dir)
    vis(FLAGS.model_variant, FLAGS.tfrecord_dir, FLAGS.dataset_split)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
