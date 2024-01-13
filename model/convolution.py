import tensorflow as tf
import tensorflow_addons as tfa

# Global residual block
class ConvolutionalBlock(tf.keras.layers.Layer):

  def __init__(self, decay, kernel_size, filters, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.decay = decay
    self.kernel_size = kernel_size
    self.filters = filters

    regularizer = tf.keras.regularizers.L2(decay)

    filter1, filter2, filter3 = filters

    self.conv_2a = tf.keras.layers.Conv2D(filter1,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 kernel_regularizer=regularizer,
                                 kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=False, padding='same')
    self.norm_2a = tfa.layers.InstanceNormalization()

    self.conv_2b = tf.keras.layers.Conv2D(filter2,
                                (kernel_size, kernel_size), strides=(2,2),
                                padding='same', kernel_regularizer=regularizer,
                                kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=False)
    self.norm_2b = tfa.layers.InstanceNormalization()

    self.conv_2c = tf.keras.layers.Conv2D(filter3, (1, 1),
                                kernel_regularizer=regularizer,
                                kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=False)
    self.norm_2c = tfa.layers.InstanceNormalization()

    self.conv_sc = tf.keras.layers.Conv2D(filter3, (1, 1),
                                strides=(2,2),
                                kernel_regularizer=regularizer,
                                kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=False)
    self.norm_sc = tfa.layers.InstanceNormalization()

  def get_config(self):
      config = super().get_config()
      config.update({"decay": self.decay,
                      "kernel_size": self.kernel_size,
                      "filters": self.filters})
      return config

  def call(self, X_input):

    activation_fn=tf.nn.relu

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
