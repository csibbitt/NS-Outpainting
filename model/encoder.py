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

    self.conv_0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4),
            strides=(2, 2),
            kernel_regularizer=self.regularizer,
            padding='same', kernel_initializer=self.initializer, use_bias=False)
    
    self.conv_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4),
            strides=(2, 2),padding='same',
            kernel_regularizer=self.regularizer,
            kernel_initializer=self.initializer, use_bias=False)

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

