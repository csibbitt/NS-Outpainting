import tensorflow as tf
import tensorflow_addons as tfa

# Skip Horizontal Connection
class Shc(tf.keras.layers.Layer):

  def __init__(self, decay, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.decay = decay
    self.regularizer = tf.keras.regularizers.L2(self.decay)

  def build_conv1(self):
    return tf.keras.layers.Conv2D(self.channels / 2, 1, strides=(1,1), activation=tf.nn.relu,
                  padding='same', use_bias=False,
                  kernel_regularizer=self.regularizer, kernel_initializer=None
    )

  def build_conv2(self):
    return tf.keras.layers.Conv2D(self.channels / 2, 3, strides=(1,1), activation=tf.nn.relu,
                  padding='same', use_bias=False,
                  kernel_regularizer=self.regularizer, kernel_initializer=None
    )

  def build_conv3(self):
    return tf.keras.layers.Conv2D(self.channels, 1, strides=(1,1), activation=None,
                  padding='same', use_bias=False,
                  kernel_regularizer=self.regularizer, kernel_initializer=None
    )

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, x, shortcut, channels):
    self.channels = channels

    x = tf.compat.v1.keras.utils.get_or_create_layer("shc_conv1_" + str(channels), self.build_conv1)(x)
    x = tf.compat.v1.keras.utils.get_or_create_layer("shc_conv1_" + str(channels) + "_in", tfa.layers.InstanceNormalization)(x)

    x = tf.compat.v1.keras.utils.get_or_create_layer("shc_conv2_" + str(channels), self.build_conv2)(x)
    x = tf.compat.v1.keras.utils.get_or_create_layer("shc_conv2_" + str(channels) + "_in", tfa.layers.InstanceNormalization)(x)

    x = tf.compat.v1.keras.utils.get_or_create_layer("shc_conv3_" + str(channels), self.build_conv3)(x)
    x = tf.compat.v1.keras.utils.get_or_create_layer("shc_conv3_" + str(channels) + "_in", tfa.layers.InstanceNormalization)(x)

    return tf.add(shortcut, x)