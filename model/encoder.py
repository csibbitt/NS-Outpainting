import tensorflow as tf
import tensorflow_addons as tfa

from model.convolution import ConvolutionalBlock
from model.identity import IdentityBlock
import model.relu as mr

# Image encoder pipeline
class Encoder(tf.keras.layers.Layer):

  def __init__(self, decay, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.initializer = tf.keras.initializers.GlorotNormal()
    self.regularizer = tf.keras.regularizers.L2(decay)
    self.convolutional_block = ConvolutionalBlock(decay)
    self.identity_block = IdentityBlock(decay, name='identity_block')

  def build_normalizer(self):
    return tfa.layers.InstanceNormalization()

  def build_conv0(self):
    return tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4),
            strides=(2, 2), name='conv0',
            kernel_regularizer=self.regularizer,
            padding='same', kernel_initializer=self.initializer, use_bias=False)

  def build_conv1(self):
    return tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4),
            strides=(2, 2), name='conv1', padding='same',
            kernel_regularizer=self.regularizer,
            kernel_initializer=self.initializer, use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, x):
    with tf.compat.v1.name_scope('tower_0/cpu_variables/model/GEN/'):
      shortcuts = []

      # stage 1
      x = tf.compat.v1.keras.utils.get_or_create_layer("main_conv0", self.build_conv0)(x)
      x = mr.in_lrelu(x, "main_act0")
      shortcuts.append(x)
      x = tf.compat.v1.keras.utils.get_or_create_layer("main_conv1", self.build_conv1)(x)
      x = mr.in_lrelu(x, "main_act1")
      shortcuts.append(x)

      # stage 2
      x = self.convolutional_block(x, kernel_size=3, filters=[
                                  64, 64, 256], stage=2, block='a', stride=2)
      x = self.identity_block(
          x, 3, [64, 64, 256], stage=2, block='b')
      x = self.identity_block(
          x, 3, [64, 64, 256], stage=2, block='c')
      shortcuts.append(x)

      # stage 3
      x = self.convolutional_block(x, kernel_size=3, filters=[128, 128, 512],
                                  stage=3, block='a', stride=2)
      x = self.identity_block(
          x, 3, [128, 128, 512], stage=3, block='b')
      x = self.identity_block(
          x, 3, [128, 128, 512], stage=3, block='c')
      x = self.identity_block(
          x, 3, [128, 128, 512], stage=3, block='d',)
      shortcuts.append(x)

      # stage 4
      x = self.convolutional_block(x, kernel_size=3, filters=[
                                  256, 256, 1024], stage=4, block='a', stride=2)
      x = self.identity_block(
          x, 3, [256, 256, 1024], stage=4, block='b')
      x = self.identity_block(
          x, 3, [256, 256, 1024], stage=4, block='c')
      x = self.identity_block(
          x, 3, [256, 256, 1024], stage=4, block='d')
      x = self.identity_block(
          x, 3, [256, 256, 1024], stage=4, block='e')
      shortcuts.append(x)

      return x, shortcuts

