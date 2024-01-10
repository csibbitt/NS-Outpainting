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

    self.conv_0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4),
            strides=(2, 2),
            kernel_regularizer=self.regularizer,
            padding='same', kernel_initializer=self.initializer, use_bias=False)

    self.conv_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4),
            strides=(2, 2),padding='same',
            kernel_regularizer=self.regularizer,
            kernel_initializer=self.initializer, use_bias=False)

    self.convolutional_block_2a = ConvolutionalBlock(decay, kernel_size=3, filters=[64, 64, 256])
    self.identity_block_2b = IdentityBlock(decay, 3, [64, 64, 256])
    self.identity_block_2c = IdentityBlock(decay, 3, [64, 64, 256])

    self.convolutional_block_3a = ConvolutionalBlock(decay, kernel_size=3, filters=[128, 128, 512])
    self.identity_block_3b = IdentityBlock(decay, 3, [128, 128, 512])
    self.identity_block_3c = IdentityBlock(decay, 3, [128, 128, 512])
    self.identity_block_3d = IdentityBlock(decay, 3, [128, 128, 512])

    self.convolutional_block_4a = ConvolutionalBlock(decay, kernel_size=3, filters=[256, 256, 1024])
    self.identity_block_4b = IdentityBlock(decay, 3, [256, 256, 1024])
    self.identity_block_4c = IdentityBlock(decay, 3, [256, 256, 1024])
    self.identity_block_4d = IdentityBlock(decay, 3, [256, 256, 1024])
    self.identity_block_4e = IdentityBlock(decay, 3, [256, 256, 1024])

  def call(self, x):
    shortcuts = []

    # stage 1
    x = self.conv_0(x)
    x = mr.in_lrelu(x, "main_act0")
    shortcuts.append(x)
    x = self.conv_1(x)
    x = mr.in_lrelu(x, "main_act1")
    shortcuts.append(x)

    # stage 2
    x = self.convolutional_block_2a(x)
    x = self.identity_block_2b(x)
    x = self.identity_block_2c(x)
    shortcuts.append(x)

    # stage 3
    x = self.convolutional_block_3a(x)
    x = self.identity_block_3b(x)
    x = self.identity_block_3c(x)
    x = self.identity_block_3d(x)
    shortcuts.append(x)

    # stage 4
    x = self.convolutional_block_4a(x)
    x = self.identity_block_4b(x)
    x = self.identity_block_4c(x)
    x = self.identity_block_4d(x)
    x = self.identity_block_4e(x)
    shortcuts.append(x)

    return x, shortcuts

