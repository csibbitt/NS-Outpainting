import tensorflow as tf
import tensorflow_addons as tfa

from model.grb import Grb
from model.identity import IdentityBlock
from model.shc import Shc


# Image encoder pipeline
class Decoder(tf.keras.layers.Layer):

  def __init__(self, decay, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.initializer = tf.keras.initializers.GlorotNormal()
    self.regularizer = tf.keras.regularizers.L2(decay)

    self.grb_t4 = Grb(decay, 1024, 1)
    self.identity_block_n4b = IdentityBlock(decay, 3, [256, 256, 1024], is_relu=True)
    self.identity_block_n4c = IdentityBlock(decay,3, [256, 256, 1024], is_relu=True)
    self.convT_4 = tf.keras.layers.Conv2DTranspose(512, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)
    self.norm_4 = tfa.layers.InstanceNormalization()
    self.shc_512 = Shc(decay, 512)

    self.grb_t3 = Grb(decay, 512, 2)
    self.identity_block_n3b = IdentityBlock(decay, 3, [128, 128, 512], is_relu=True)
    self.identity_block_n3c = IdentityBlock(decay, 3, [128, 128, 512], is_relu=True)
    self.identity_block_n3d = IdentityBlock(decay, 3, [128, 128, 512], is_relu=True)
    self.convT_3 = tf.keras.layers.Conv2DTranspose(256, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)
    self.norm_3 = tfa.layers.InstanceNormalization()
    self.shc_256 = Shc(decay, 256)

    self.grb_t2 = Grb(decay, 256, 4)
    self.identity_block_n2b = IdentityBlock(decay, 3, [64, 64, 256], is_relu=True)
    self.identity_block_n2c = IdentityBlock(decay, 3, [64, 64, 256], is_relu=True)
    self.identity_block_n2d = IdentityBlock(decay, 3, [64, 64, 256], is_relu=True)
    self.identity_block_n2e = IdentityBlock(decay, 3, [64, 64, 256], is_relu=True)
    self.convT_2 = tf.keras.layers.Conv2DTranspose(128, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)
    self.norm_2 = tfa.layers.InstanceNormalization()
    self.shc_128 = Shc(decay, 128)

    self.convT_1 = tf.keras.layers.Conv2DTranspose(64, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)
    self.norm_1 = tfa.layers.InstanceNormalization()
    self.shc_64 = Shc(decay, 64)

    self.convT_0 = tf.keras.layers.Conv2DTranspose(3, 4, strides=(2,2),
              activation=None, padding='same', kernel_initializer=self.initializer,
              kernel_regularizer=self.regularizer,
              bias_initializer=None, use_bias=False)

  def call(self, x, shortcuts):
    # stage -4
    x = tf.concat([shortcuts[4], x], axis=2)

    x = self.grb_t4(x)
    x = self.identity_block_n4b(x)
    x = self.identity_block_n4c(x)
    x = self.convT_4(x)
    x = self.norm_4(x)
    sc, kp = tf.split(x, 2, axis=2)
    sc = tf.nn.relu(sc)
    merge = tf.concat([shortcuts[3], sc], axis=3)
    merge = self.shc_512(merge, shortcuts[3])
    merge = mr.in_relu(merge, "main_actT4_merge")
    x = tf.concat(
        [merge, kp], axis=2)


    # stage -3
    x = self.grb_t3(x)
    x = self.identity_block_n3b(x)
    x = self.identity_block_n3c(x)
    x = self.identity_block_n3d(x)

    x = self.convT_3(x)
    x = self.norm_3(x)
    sc, kp = tf.split(x, 2, axis=2)
    sc = tf.nn.relu(sc)
    merge = tf.concat([shortcuts[2], sc], axis=3)
    merge = self.shc_256(merge, shortcuts[2])
    merge = mr.in_relu(merge, "main_actT3_merge")
    x = tf.concat(
        [merge, kp], axis=2)

    # stage -2
    x = self.grb_t2(x)
    x = self.identity_block_n2b(x)
    x = self.identity_block_n2c(x)
    x = self.identity_block_n2d(x)
    x = self.identity_block_n2e(x)

    x = self.convT_2(x)
    x = self.norm_2(x)
    sc, kp = tf.split(x, 2, axis=2)
    sc = tf.nn.relu(sc)
    merge = tf.concat([shortcuts[1], sc], axis=3)
    merge = self.shc_128(merge, shortcuts[1])
    merge = mr.in_relu(merge, "main_actT2_merge")
    x = tf.concat(
        [merge, kp], axis=2)

    # stage -1

    x = self.convT_1(x)
    x = self.norm_1(x)
    sc, kp = tf.split(x, 2, axis=2)
    sc = tf.nn.relu(sc)
    merge = tf.concat([shortcuts[0], sc], axis=3)
    merge = self.shc_64(merge, shortcuts[0])
    merge = mr.in_relu(merge, "main_actT1_merge")
    x = tf.concat(
        [merge, kp], axis=2)

    # stage -0
    recon = self.convT_0(x)

    return recon
