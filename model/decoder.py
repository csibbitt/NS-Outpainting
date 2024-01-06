import tensorflow as tf
import tensorflow_addons as tfa

from model.grb import Grb
from model.identity import IdentityBlock
import model.relu as mr
from model.shc import Shc


# Image encoder pipeline
class Decoder(tf.keras.layers.Layer):

  def __init__(self, decay, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.initializer = tf.compat.v1.keras.initializers.glorot_normal()
    self.regularizer = tf.keras.regularizers.L2(decay)
    self.grb = Grb(decay)
    self.identity_block = IdentityBlock(decay)
    self.shc = Shc(decay)

  def build_normalizer(self):
      return tfa.layers.InstanceNormalization()

  def build_convT4(self):
      return tf.keras.layers.Conv2DTranspose(512, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)

  def build_convT3(self):
      return tf.keras.layers.Conv2DTranspose(256, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)

  def build_convT2(self):
      return tf.keras.layers.Conv2DTranspose(128, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)

  def build_convT1(self):
      return tf.keras.layers.Conv2DTranspose(64, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)

  def build_convT0(self):
      return tf.keras.layers.Conv2DTranspose(3, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, x, shortcuts):

    # stage -4
    x = tf.concat([shortcuts[4], x], axis=2)

    x = self.grb(x, 1024, 1, 't4')
    x = self.identity_block(
        x, 3, [256, 256, 1024], stage=-4, block='b', is_relu=True)
    x = self.identity_block(
        x, 3, [256, 256, 1024], stage=-4, block='c', is_relu=True)


    x = tf.compat.v1.keras.utils.get_or_create_layer("main_convT4", self.build_convT4)(x)
    x = tf.compat.v1.keras.utils.get_or_create_layer("main_convT4_in", self.build_normalizer)(x)
    sc, kp = tf.split(x, 2, axis=2)
    sc = tf.nn.relu(sc)
    merge = tf.concat([shortcuts[3], sc], axis=3)
    merge = self.shc(merge, shortcuts[3], 512)
    merge = mr.in_relu(merge, "main_actT4_merge")
    x = tf.concat(
        [merge, kp], axis=2)


    # stage -3
    x = self.grb(x, 512, 2, 't3')
    x = self.identity_block(
        x, 3, [128, 128, 512], stage=-3, block='b', is_relu=True)
    x = self.identity_block(
        x, 3, [128, 128, 512], stage=-3, block='c', is_relu=True)
    x = self.identity_block(
        x, 3, [128, 128, 512], stage=-3, block='d', is_relu=True)

    x = tf.compat.v1.keras.utils.get_or_create_layer("main_convT3", self.build_convT3)(x)
    x = tf.compat.v1.keras.utils.get_or_create_layer("main_convT3_in", self.build_normalizer)(x)
    sc, kp = tf.split(x, 2, axis=2)
    sc = tf.nn.relu(sc)
    merge = tf.concat([shortcuts[2], sc], axis=3)
    merge = self.shc(merge, shortcuts[2], 256)
    merge = mr.in_relu(merge, "main_actT3_merge")
    x = tf.concat(
        [merge, kp], axis=2)

    # stage -2
    x = self.grb(x, 256, 4, 't2')
    x = self.identity_block(
        x, 3, [64, 64, 256], stage=-2, block='b', is_relu=True)
    x = self.identity_block(
        x, 3, [64, 64, 256], stage=-2, block='c', is_relu=True)
    x = self.identity_block(
        x, 3, [64, 64, 256], stage=-2, block='d', is_relu=True)
    x = self.identity_block(
        x, 3, [64, 64, 256], stage=-2, block='e', is_relu=True)

    x = tf.compat.v1.keras.utils.get_or_create_layer("main_convT2", self.build_convT2)(x)
    x = tf.compat.v1.keras.utils.get_or_create_layer("main_convT2_in", self.build_normalizer)(x)
    sc, kp = tf.split(x, 2, axis=2)
    sc = tf.nn.relu(sc)
    merge = tf.concat([shortcuts[1], sc], axis=3)
    merge = self.shc(merge, shortcuts[1], 128)
    merge = mr.in_relu(merge, "main_actT2_merge")
    x = tf.concat(
        [merge, kp], axis=2)

    # stage -1

    x = tf.compat.v1.keras.utils.get_or_create_layer("main_convT1", self.build_convT1)(x)
    x = tf.compat.v1.keras.utils.get_or_create_layer("main_convT1_in", self.build_normalizer)(x)
    sc, kp = tf.split(x, 2, axis=2)
    sc = tf.nn.relu(sc)
    merge = tf.concat([shortcuts[0], sc], axis=3)
    merge = self.shc(merge, shortcuts[0], 64)
    merge = mr.in_relu(merge, "main_actT1_merge")
    x = tf.concat(
        [merge, kp], axis=2)

    # stage -0
    recon = tf.compat.v1.keras.utils.get_or_create_layer("main_convT0", self.build_convT0)(x)

    return recon