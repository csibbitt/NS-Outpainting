import tensorflow as tf
import tensorflow_addons as tfa

# Global residual block
class Grb(tf.keras.layers.Layer):

  def __init__(self, decay, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.activation_fn = tf.nn.relu
    self.normalizer_fn = tfa.layers.InstanceNormalization
    self.weight_decay = decay

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def new_atrous_conv_layer(self, bottom, filter_shape, rate, name=None):
      with tf.compat.v1.variable_scope(name):
          regularizer = tf.keras.regularizers.L2(self.weight_decay)
          initializer = tf.compat.v1.keras.initializers.glorot_normal()
          W = tf.compat.v1.get_variable(
              "W",
              shape=filter_shape,
              regularizer=regularizer,
              initializer=initializer)

          x = tf.nn.atrous_conv2d(
              bottom, W, rate, padding='SAME')
      return x

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, x, filters, rate, name):
    shortcut = x
    x1 = self.new_atrous_conv_layer(x, [3, 1, filters, filters], rate, name+'_a1')
    x1 = self.normalizer_fn()(x1)
    x1 = self.activation_fn(x1)
    x1 = self.new_atrous_conv_layer(x1, [1, 7, filters, filters], rate, name+'_a2')
    x1 = self.normalizer_fn()(x1)

    x2 = self.new_atrous_conv_layer(x, [1, 7, filters, filters], rate, name+'_b1')
    x2 = self.normalizer_fn()(x2)
    x2 = self.activation_fn(x2)
    x2 = self.new_atrous_conv_layer(x2, [3, 1, filters, filters], rate, name+'_b2')
    x2 = self.normalizer_fn()(x2)

    x = tf.add(shortcut, x1)
    x = tf.add(x, x2)
    x = self.activation_fn(x)
    return x

