from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow import keras
import input_pipeline

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_split', 'train', 'Using which dataset split to train the network.')
flags.DEFINE_integer('train_batch_size', 32, 'Batch size used for train.')
flags.DEFINE_integer('val_batch_size', 32, 'Batch size used for train.')
flags.DEFINE_boolean('is_training', True, 'Is training?')
flags.DEFINE_integer('num_train_epochs', 10, 'Number training epochs.')


# model 
def define_model():
    inputs = keras.Input(shape=input_pipeline.IMAGE_SHAPE)

    # stage 1
    x = keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu)(inputs)
    x = keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu)(x)
    x = keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')(x)

    # stage 2
    x = keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu)(x)
    x = keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu)(x)
    x = keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')(x)

    # stage 3
    x = keras.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu)(x)
    x = keras.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu)(x)
    x = keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')(x)

    # stage 4
    x = keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu)(x)
    x = keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu)(x)
    x = keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(64, activation=tf.nn.relu)(x)

    predictions = keras.layers.Dense(8, activation=None)(x)

    model = keras.Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])
    
    return model


def main(unused_argv):
    with tf.device('/cpu:0'):
        train_dataset = input_pipeline.inputs('train', is_training=True, 
            batch_size=FLAGS.train_batch_size, num_epochs=None)
        val_dataset = input_pipeline.inputs('val', is_training=False, 
            batch_size=FLAGS.val_batch_size, num_epochs=None)
    
    model = define_model()

    steps_per_epoch = np.int32(np.ceil(input_pipeline.NUMBER_TRAIN_DATA / FLAGS.train_batch_size))
    validation_steps = np.int32(np.ceil(input_pipeline.NUMBER_VAL_DATA / FLAGS.val_batch_size))

    model.fit(train_dataset, epochs=FLAGS.num_train_epochs, steps_per_epoch=steps_per_epoch,
              validation_data=val_dataset, validation_steps=validation_steps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()