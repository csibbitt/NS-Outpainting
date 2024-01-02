import tensorflow as tf
import tensorflow_addons as tfa

import modeling.relu as mr

# Global residual block
class ConvolutionalBlock(tf.keras.layers.Layer):

  def __init__(self, decay, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.decay = decay
    self.initializer = tf.compat.v1.keras.initializers.glorot_normal()

  def build_normalizer(self):
    return tfa.layers.InstanceNormalization()

  def build_regularizer(self):
    return tf.keras.regularizers.L2(self.decay)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def build_conv1(self):
    return tf.keras.layers.Conv2D(self.filter1,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 name=self.conv_name_base + '2a',
                                 kernel_regularizer=tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "conv_regularizer2a", self.build_regularizer),
                                 kernel_initializer=self.initializer, use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def build_conv2(self):
    return tf.keras.layers.Conv2D(self.filter2,
                                (self.kernel_size, self.kernel_size), strides=(self.stride, self.stride), 
                                name=self.conv_name_base +'2b',
                                padding='same', kernel_regularizer=tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_regularizer2b", self.build_regularizer),
                                kernel_initializer=self.initializer, use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def build_conv3(self):
    return tf.keras.layers.Conv2D(self.filter3, (1, 1),
                                name=self.conv_name_base + '2c',
                                kernel_regularizer=tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_regularizer2c", self.build_regularizer),
                                kernel_initializer=self.initializer, use_bias=False)

  def build_conv_short(self):
    return tf.keras.layers.Conv2D(self.filter3, (1, 1),
                                strides=(self.stride, self.stride),
                                name=self.conv_name_base + '1',
                                kernel_regularizer=tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_regularizer1", self.build_regularizer),
                                kernel_initializer=self.initializer, use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, X_input, kernel_size, filters, stage, block, stride=2, is_relu=False):

    self.conv_name_base = 'res' + str(stage) + block + '_branch'
    self.kernel_size = kernel_size
    self.stage = stage
    self.block = block
    self.stride = stride

    self.filter1, self.filter2, self.filter3 = filters

    if is_relu:
        activation_fn=tf.nn.relu
    else:
        activation_fn=mr.leaky_relu

    with tf.compat.v1.variable_scope("conv_block_stage" + str(stage) + block):
      X_shortcut = X_input

      # First component of main path
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_conv2a", self.build_conv1)(X_input)
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_nomalizer2a", self.build_normalizer)(x)
      x = activation_fn(x)

      # Second component of main path
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_conv2b", self.build_conv2)(x)
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_nomalizer2b", self.build_normalizer)(x)
      x = activation_fn(x)

      # Third component of main path
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_conv2c", self.build_conv3)(x)
      x = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_nomalizer2c", self.build_normalizer)(x)

      # SHORTCUT PATH
      X_shortcut = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_conv1", self.build_conv_short)(X_shortcut)
      X_shortcut = tf.compat.v1.keras.utils.get_or_create_layer(self.conv_name_base + "_conv_nomalizer1", self.build_normalizer)(X_shortcut)

      # Final step: Add shortcut value to main path, and pass it through
      x = tf.add(x, X_shortcut)
      x = activation_fn(x)

    return x