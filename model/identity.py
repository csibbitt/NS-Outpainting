import tensorflow as tf
import tensorflow_addons as tfa

import model.relu as mr

# Identity Block
class IdentityBlock(tf.keras.layers.Layer):

  def __init__(self, decay, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.decay = decay
    self.initializer = tf.keras.initializers.GlorotNormal()
    self.regularizer = tf.keras.regularizers.L2(self.decay)

    self.conv_1 = tf.keras.layers.Conv2D(self.filter1,
                                kernel_size=(1, 1), strides=(1, 1),
                                name=self.conv_name_base + '2a',
                                kernel_regularizer=self.regularizer,
                                kernel_initializer=self.initializer, use_bias=False, padding='same')
    self.norm_1 = tfa.layers.InstanceNormalization()

    self.conv_2 = tf.keras.layers.Conv2D(self.filter2,
                                (self.kernel_size, self.kernel_size),
                                padding='same', name=self.conv_name_base + '2b',
                                kernel_regularizer=self.regularizer,
                                kernel_initializer=self.initializer, use_bias=False)
    self.norm_2 = tfa.layers.InstanceNormalization()

    self.conv_3 = tf.keras.layers.Conv2D(self.filter3,
                                kernel_size=(1, 1), name=self.conv_name_base + '2c',
                                kernel_regularizer=self.regularizer,
                                kernel_initializer=self.initializer, use_bias=False)
    self.norm_4 = tfa.layers.InstanceNormalization()

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

    X_shortcut = X_input

    # First component of main path
    x = self.conv_1(X_input)
    x = self.norm_1(x)
    x = activation_fn(x)

    # Second component of main path
    x = self.conv_2(x)
    x = self.norm_2(x)
    x = activation_fn(x)

    # Third component of main path
    x = self.conv_3(x)
    x = self.norm_4(x)

    # Final step: Add shortcut value to main path, and pass it through
    x = tf.add(x, X_shortcut)
    x = activation_fn(x)

    return x
