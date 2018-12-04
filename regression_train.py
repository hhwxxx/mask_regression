from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import sys
sys.path.extend(['/home/hhw/work/hikvision', '/home/hhw/work/hikvision/slim'])

import tensorflow as tf
import numpy as np

import input_pipeline
import core
import models


slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_dir', 
                    './tfrecords/quadrilateral_multiple', 
                    'Directory containing tfrecords.')
flags.DEFINE_string('dataset_split', 'train', 
                    'Using which dataset split to train the network.')
flags.DEFINE_string('model_variant', 'vgg_16', 'Model to train.')
flags.DEFINE_string('restore_ckpt_path', 
                    './init_models/vgg_16.ckpt', 
                    'Path to restore checkpoint.')
flags.DEFINE_string('train_dir',
                    './exp/vgg_quadrilateral_multiple_01/train', 
                    'Training directory.')

flags.DEFINE_multi_integer('weights', [1, 1],
                           'Weights used to compute loss.')
flags.DEFINE_integer('batch_size', 8, 'Batch size used for train.')
flags.DEFINE_boolean('is_training', True, 'Is training?')

flags.DEFINE_integer('decay_epochs', 90, 
                     'Decay steps in exponential learning rate decay policy.')
flags.DEFINE_integer('num_epochs', 150, 'Number epochs.')
flags.DEFINE_float('initial_learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.1, 
                   'Decay rate in exponential learning rate decay policy.')
flags.DEFINE_boolean('staircase', True, 'Staircase?')

flags.DEFINE_integer('save_checkpoint_steps', 500, 'Save checkpoint steps.')
flags.DEFINE_integer('log_frequency', 10, 'Log frequency.')


def train(model_variant, tfrecord_dir, dataset_split):
    with tf.Graph().as_default() as g:
        global_step  = tf.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            images, labels = input_pipeline.inputs(
                tfrecord_dir, dataset_split, FLAGS.is_training,
                FLAGS.batch_size, num_epochs=None)
        
        predictions = core.inference(model_variant, images,
                                     is_training=FLAGS.is_training)

        total_loss = core.loss(predictions, labels,
                               FLAGS.weights)

        # metric
        mean_absolute_error, update_op = tf.metrics.mean_absolute_error(
            labels=labels, predictions=predictions, 
            updates_collections=tf.GraphKeys.UPDATE_OPS, 
            name='mean_absolute_error')
        tf.summary.scalar('train/mean_absolute_error', update_op)

        steps_per_epoch = np.ceil(
            input_pipeline.NUMBER_TRAIN_DATA / FLAGS.batch_size)
        decay_steps = FLAGS.decay_epochs * steps_per_epoch

        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate, global_step, decay_steps,
            FLAGS.decay_rate, staircase=FLAGS.staircase)
        tf.summary.scalar('learning_rate', learning_rate)

        with tf.variable_scope('adam_vars'): 
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # update moving mean/var in batch_norm layers
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(total_loss, global_step)


        adam_vars = optimizer.variables()
        # def name_in_checkpoint(var):
        #     return var.op.name.replace(FLAGS.model_variant, 'vgg_16')
        variables_to_restore = slim.get_variables_to_restore(
            exclude=(models.EXCLUDE_LIST_MAP[FLAGS.model_variant]
                     + ['global_step', 'adam_vars']))
        # variables_to_restore = {name_in_checkpoint(var):var
        #     for var in variables_to_restore
        #         if not 'BatchNorm' in var.op.name}
        # variables_to_restore = {name_in_checkpoint(var):var
        #     for var in variables_to_restore}

        restorer = tf.train.Saver(variables_to_restore)
        def init_fn(scaffold, sess):
            restorer.restore(sess, FLAGS.restore_ckpt_path)


        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                # Assuming model_checkpoint_path looks something like:                                                                                                                             
                #   /my-favorite-path/cifar10_train/model.ckpt-0,                                                                                                                                  
                # extract global_step from it.             
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    self._step = int(
                        ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) - 1
                else:
                    self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(total_loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = (FLAGS.log_frequency
                                        * FLAGS.batch_size / duration)
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.5f '
                                  '(%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value, 
                                        examples_per_sec, sec_per_batch))


        training_steps = steps_per_epoch * FLAGS.num_epochs

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            scaffold=tf.train.Scaffold(init_fn=init_fn),
            hooks=[tf.train.StopAtStepHook(last_step=training_steps),
                   tf.train.NanTensorHook(total_loss),
                   _LoggerHook()],
            config=config,
            save_checkpoint_steps=FLAGS.save_checkpoint_steps) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(unused_argv):
    if not os.path.exists(FLAGS.train_dir): 
        os.makedirs(FLAGS.train_dir)
    train(FLAGS.model_variant, FLAGS.tfrecord_dir, FLAGS.dataset_split)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
