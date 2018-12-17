from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import sys

import tensorflow as tf

import input_pipeline
import core


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_dir', 
                    './tfrecords/quadrilateral_multiple', 
                    'Directory containing tfrecords.')
flags.DEFINE_string('dataset_split', 'val', 'Dataset split used to evaluate.')
flags.DEFINE_string('model_variant', 'vgg_16', 'Model variant.')
flags.DEFINE_string('checkpoint_dir',
                    './exp/vgg_quadrilateral_multiple_01/train',
                    'Directory containing trained checkpoints.')
flags.DEFINE_string('eval_dir', 
                    './exp/vgg_quadrilateral_multiple_01/eval', 
                    'Evaluation directory.')

flags.DEFINE_integer('batch_size', 2, 'Batch size.')
flags.DEFINE_boolean('is_training', False, 'Is training?')
flags.DEFINE_integer('eval_interval_secs', 60 * 2, 
                     'Evaluation interval seconds.')


def eval(model_variant, tfrecord_dir, dataset_split):
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            images, labels = input_pipeline.inputs(
                tfrecord_dir, dataset_split, FLAGS.is_training,
                FLAGS.batch_size, num_epochs=1)
        predictions = core.inference(model_variant, images, FLAGS.is_training)

        mean_absolute_error, update_op = tf.metrics.mean_absolute_error(
            labels=labels, predictions=predictions,
            name='mean_absolute_error')

        summary_op = tf.summary.scalar(
            'eval/mean_absolute_error', mean_absolute_error)
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        num_batches = int(
            math.ceil(input_pipeline.NUMBER_VAL_DATA / FLAGS.batch_size))

        # get global_step used in summary_writer.
        ckpt = tf.train.get_checkpoint_state(
            checkpoint_dir=FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = int(
                ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print('Get global_step from checkpoint name.')
        else:
            global_step = tf.train.get_or_create_global_step()
            print('Create gloabl_step')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                config=config,
                checkpoint_dir=FLAGS.checkpoint_dir)) as mon_sess:
            for _ in range(num_batches):
                mon_sess.run(update_op)

            summary = mon_sess.run(summary_op)
            summary_writer.add_summary(summary, global_step=global_step)
            summary_writer.flush()
            print('*' * 50)
            print('Step {:06} mean_absolute_error:'.format(global_step), 
                  mon_sess.run(mean_absolute_error))
            print('*' * 50)
            summary_writer.close()


def main(unused_argv):
    while True:
        eval(FLAGS.model_variant, FLAGS.tfrecord_dir, FLAGS.dataset_split)
        time.sleep(FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
