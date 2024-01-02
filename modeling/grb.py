import tensorflow as tf
import tensorflow_addons as tfa

# Global residual block
class Grb(tf.keras.layers.Layer):

  def __init__(self, decay, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.decay = decay
    self.activation_fn = tf.nn.relu
  
  def build_normalizer(self):
    return tfa.layers.InstanceNormalization()

  def build_regularizer(self):
    return tf.keras.regularizers.L2(self.decay)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def new_atrous_conv_layer(self, bottom, filter_shape, rate, name=None):
      with tf.compat.v1.variable_scope(name):
          initializer = tf.compat.v1.keras.initializers.glorot_normal()
          W = tf.compat.v1.get_variable(
              "W",
              shape=filter_shape,
              regularizer=tf.compat.v1.keras.utils.get_or_create_layer("grb_regularizer", self.build_regularizer),
              initializer=initializer)

          x = tf.nn.atrous_conv2d(
              bottom, W, rate, padding='SAME')
      return x

  def call(self, x, filters, rate, name):
    shortcut = x
    x1 = self.new_atrous_conv_layer(x, [3, 1, filters, filters], rate, name+'_a1')
    x1 = tf.compat.v1.keras.utils.get_or_create_layer("grb_nomalizer1", self.build_normalizer)(x1)
    x1 = self.activation_fn(x1)
    x1 = self.new_atrous_conv_layer(x1, [1, 7, filters, filters], rate, name+'_a2')
    x1 = tf.compat.v1.keras.utils.get_or_create_layer("grb_nomalizer2", self.build_normalizer)(x1)

    x2 = self.new_atrous_conv_layer(x, [1, 7, filters, filters], rate, name+'_b1')
    x2 = tf.compat.v1.keras.utils.get_or_create_layer("grb_nomalizer3", self.build_normalizer)(x2)
    x2 = self.activation_fn(x2)
    x2 = self.new_atrous_conv_layer(x2, [3, 1, filters, filters], rate, name+'_b2')
    x2 = tf.compat.v1.keras.utils.get_or_create_layer("grb_nomalizer4", self.build_normalizer)(x2)

    x = tf.add(shortcut, x1)
    x = tf.add(x, x2)
    x = self.activation_fn(x)
    return x

