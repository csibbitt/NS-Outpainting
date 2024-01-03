import tensorflow as tf
import tensorflow_addons as tfa

import modeling.relu as mr

# Global residual block
class IdentityBlock(tf.keras.layers.Layer):

  def __init__(self, decay, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.decay = decay
    self.initializer = tf.compat.v1.keras.initializers.glorot_normal(seed=4)

  def build_normalizer(self):
    return tfa.layers.InstanceNormalization()

  def build_regularizer(self):
    return tf.keras.regularizers.L2(self.decay)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def build_conv1(self):
    return tf.keras.layers.Conv2D(self.filter1,
                                kernel_size=(1, 1), strides=(1, 1),
                                name=self.conv_name_base + '2a',
                                kernel_regularizer=tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_id_regularizer1", self.build_regularizer),
                                kernel_initializer=self.initializer, use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def build_conv2(self):
    return tf.keras.layers.Conv2D(self.filter2,
                                (self.kernel_size, self.kernel_size),
                                padding='same', name=self.conv_name_base + '2b',
                                kernel_regularizer=tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_id_regularizer2", self.build_regularizer),
                                kernel_initializer=self.initializer, use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def build_conv3(self):
    return tf.keras.layers.Conv2D(self.filter3,
                                kernel_size=(1, 1), name=self.conv_name_base + '2c',
                                kernel_regularizer=tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_id_regularizer3", self.build_regularizer),
                                kernel_initializer=self.initializer, use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, X_input, kernel_size, filters, stage, block, is_relu=False):

    self.conv_name_base = 'res' + str(stage) + block + '_branch'
    self.kernel_size = kernel_size
    self.stage = stage
    self.block = block

    self.filter1, self.filter2, self.filter3 = filters

    if is_relu:
      activation_fn=tf.nn.relu
    else:
      activation_fn=mr.leaky_relu

    with tf.compat.v1.variable_scope("id_block_stage" + str(self.stage) + self.block):
      X_shortcut = X_input


      # First component of main path
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_id_conv1", self.build_conv1)(X_input)
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_id_nomalizer1", self.build_normalizer)(x)
      x = activation_fn(x, self.conv_name_base + "_id_act1")

      # Second component of main path
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_id_conv2", self.build_conv2)(x)
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_id_nomalizer2", self.build_normalizer)(x)
      x = activation_fn(x, self.conv_name_base + "_id_act2")

      # Third component of main path
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_id_conv3", self.build_conv3)(x)
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_id_nomalizer3", self.build_normalizer)(x)

      # Final step: Add shortcut value to main path, and pass it through
      x = tf.add(x, X_shortcut)
      x = activation_fn(x, self.conv_name_base + "_id_act3")

    return x
