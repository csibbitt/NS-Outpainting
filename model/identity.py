import tensorflow as tf
import tensorflow_addons as tfa

import model.relu as mr

# Identity Block
class IdentityBlock(tf.keras.layers.Layer):

  def __init__(self, decay, kernel_size, filters, is_relu=False, *args, **kwargs):
    super().__init__(*args, **kwargs)

    filter1, filter2, filter3 = filters

    self.decay = decay
    self.initializer = tf.keras.initializers.GlorotNormal()
    self.regularizer = tf.keras.regularizers.L2(self.decay)

    if is_relu:
      self.activation_fn=tf.nn.relu
    else:
      self.activation_fn=mr.leaky_relu

    self.conv_1 = tf.keras.layers.Conv2D(filter1,
                                kernel_size=(1, 1), strides=(1, 1),
                                name=self.conv_name_base + '2a',
                                kernel_regularizer=self.regularizer,
                                kernel_initializer=self.initializer, use_bias=False, padding='same')
    self.norm_1 = tfa.layers.InstanceNormalization()

    self.conv_2 = tf.keras.layers.Conv2D(filter2,
                                (kernel_size, kernel_size),
                                padding='same', name=self.conv_name_base + '2b',
                                kernel_regularizer=self.regularizer,
                                kernel_initializer=self.initializer, use_bias=False)
    self.norm_2 = tfa.layers.InstanceNormalization()

    self.conv_3 = tf.keras.layers.Conv2D(filter3,
                                kernel_size=(1, 1), name=self.conv_name_base + '2c',
                                kernel_regularizer=self.regularizer,
                                kernel_initializer=self.initializer, use_bias=False)
    self.norm_4 = tfa.layers.InstanceNormalization()

  def call(self, X_input):

    X_shortcut = X_input

    # First component of main path
    x = self.conv_1(X_input)
    x = self.norm_1(x)
    x = self.activation_fn(x)

    # Second component of main path
    x = self.conv_2(x)
    x = self.norm_2(x)
    x = self.activation_fn(x)

    # Third component of main path
    x = self.conv_3(x)
    x = self.norm_4(x)

    # Final step: Add shortcut value to main path, and pass it through
    x = tf.add(x, X_shortcut)
    x = self.activation_fn(x)

    return x
