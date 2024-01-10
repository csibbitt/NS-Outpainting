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
    self.initializer = tf.keras.initializers.GlorotNormal()
    self.regularizer = tf.keras.regularizers.L2(decay)
    self.grb = Grb(decay)
    self.identity_block = IdentityBlock(decay, name='identity_block')
    self.shc = Shc(decay)

    self.convT_4 = tf.keras.layers.Conv2DTranspose(512, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)
    self.norm_4 = tfa.layers.InstanceNormalization()

    self.convT_3 = tf.keras.layers.Conv2DTranspose(256, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)
    self.norm_3 = tfa.layers.InstanceNormalization()

    self.convT_2 = tf.keras.layers.Conv2DTranspose(128, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)
    self.norm_2 = tfa.layers.InstanceNormalization()

    self.convT_1 = tf.keras.layers.Conv2DTranspose(64, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)
    self.norm_1 = tfa.layers.InstanceNormalization()

    self.convT_0 = tf.keras.layers.Conv2DTranspose(3, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)

  def call(self, x, shortcuts):
    # stage -4
    x = tf.concat([shortcuts[4], x], axis=2)

    x = self.grb(x, 1024, 1, 't4')
    x = self.identity_block(
        x, 3, [256, 256, 1024], stage=-4, block='b', is_relu=True)
    x = self.identity_block(
        x, 3, [256, 256, 1024], stage=-4, block='c', is_relu=True)

    x = self.convT_4(x)
    x = self.norm_4(x)
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

    x = self.convT_3(x)
    x = self.norm_3(x)
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

    x = self.convT_2(x)
    x = self.norm_2(x)
    sc, kp = tf.split(x, 2, axis=2)
    sc = tf.nn.relu(sc)
    merge = tf.concat([shortcuts[1], sc], axis=3)
    merge = self.shc(merge, shortcuts[1], 128)
    merge = mr.in_relu(merge, "main_actT2_merge")
    x = tf.concat(
        [merge, kp], axis=2)

    # stage -1

    x = self.convT_1(x)
    x = self.norm_1(x)
    sc, kp = tf.split(x, 2, axis=2)
    sc = tf.nn.relu(sc)
    merge = tf.concat([shortcuts[0], sc], axis=3)
    merge = self.shc(merge, shortcuts[0], 64)
    merge = mr.in_relu(merge, "main_actT1_merge")
    x = tf.concat(
        [merge, kp], axis=2)

    # stage -0
    recon = self.convT_0(x)

    return recon
