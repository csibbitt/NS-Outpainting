import tensorflow as tf
import tensorflow_addons as tfa

# Skip Horizontal Connection
class Shc(tf.keras.layers.Layer):

  def __init__(self, decay, channels, *args, **kwargs):
    super().__init__(*args, **kwargs)

    regularizer = tf.keras.regularizers.L2(decay)

    self.conv_1 = tf.keras.layers.Conv2D(channels / 2, 1, strides=(1,1), activation=tf.nn.relu,
                  padding='same', use_bias=False,
                  kernel_regularizer=regularizer, kernel_initializer=None)
    self.norm_1 = tfa.layers.InstanceNormalization()

    self.conv_2 = tf.keras.layers.Conv2D(channels / 2, 3, strides=(1,1), activation=tf.nn.relu,
                  padding='same', use_bias=False,
                  kernel_regularizer=regularizer, kernel_initializer=None)
    self.norm_2 = tfa.layers.InstanceNormalization()

    self.conv_3 = tf.keras.layers.Conv2D(channels, 1, strides=(1,1), activation=None,
                  padding='same', use_bias=False,
                  kernel_regularizer=regularizer, kernel_initializer=None)
    self.norm_3 = tfa.layers.InstanceNormalization()

  def call(self, x, shortcut):

    x = self.conv_1(x)
    x = self.norm_1(x)

    x = self.conv_2(x)
    x = self.norm_2(x)

    x = self.conv_3(x)
    x = self.norm_3(x)

    return tf.add(shortcut, x)