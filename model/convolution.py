import tensorflow as tf
import tensorflow_addons as tfa

import model.relu as mr

# Global residual block
class ConvolutionalBlock(tf.keras.layers.Layer):

  def __init__(self, decay, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.decay = decay
    self.initializer = tf.keras.initializers.GlorotNormal()
    self.regularizer = tf.keras.regularizers.L2(self.decay)

    self.conv_2a = tf.keras.layers.Conv2D(self.filter1,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 kernel_regularizer=self.regularizer,
                                 kernel_initializer=self.initializer, use_bias=False, padding='same') #** Original code had no padding here?
    self.norm_2a = tfa.layers.InstanceNormalization()

    self.conv_2b = tf.keras.layers.Conv2D(self.filter2,
                                (self.kernel_size, self.kernel_size), strides=(self.stride, self.stride), 
                                padding='same', kernel_regularizer=self.regularizer,
                                kernel_initializer=self.initializer, use_bias=False)
    self.norm_2b = tfa.layers.InstanceNormalization()

    self.conv_2c = tf.keras.layers.Conv2D(self.filter3, (1, 1),
                                kernel_regularizer=self.regularizer,
                                kernel_initializer=self.initializer, use_bias=False)
    self.norm_2c = tfa.layers.InstanceNormalization()

    self.conv_sc = tf.keras.layers.Conv2D(self.filter3, (1, 1),
                                strides=(self.stride, self.stride),
                                kernel_regularizer=self.regularizer,
                                kernel_initializer=self.initializer, use_bias=False)
    self.norm_sc = tfa.layers.InstanceNormalization()

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

    X_shortcut = X_input

    x = self.conv_2a(X_input)
    x = self.norm_2a(x)
    x = activation_fn(x)

    x = self.conv_2b(x)
    x = self.norm_2b(x)
    x = activation_fn(x)

    x = self.conv_2c(x)
    x = self.norm_2c(x)

    # SHORTCUT PATH
    X_shortcut = self.conv_sc(X_shortcut)
    X_shortcut = self.norm_sc(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through
    x = tf.add(x, X_shortcut)
    x = activation_fn(x)

    return x
