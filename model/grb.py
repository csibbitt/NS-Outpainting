import tensorflow as tf
import tensorflow_addons as tfa

# Global residual block
class Grb(tf.keras.layers.Layer):

  def __init__(self, decay, filters, rate, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.decay = decay
    self.filters = filters

    self.atrous_a1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,1), dilation_rate=rate,
                                            kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                            kernel_regularizer=tf.keras.regularizers.L2(decay),
                                            padding='SAME')
    self.norm_a1 = tfa.layers.InstanceNormalization()

    self.atrous_a2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,7), dilation_rate=rate,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        kernel_regularizer=tf.keras.regularizers.L2(decay),
                                        padding='SAME')
    self.norm_a2 = tfa.layers.InstanceNormalization()

    self.atrous_b1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,7), dilation_rate=rate,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        kernel_regularizer=tf.keras.regularizers.L2(decay),
                                        padding='SAME')
    self.norm_b1 = tfa.layers.InstanceNormalization()

    self.atrous_b2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,1), dilation_rate=rate,
                                            kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                            kernel_regularizer=tf.keras.regularizers.L2(decay),
                                            padding='SAME')
    self.norm_b2 = tfa.layers.InstanceNormalization()

  def get_config(self):
      config = super().get_config()
      config.update({"decay": self.decay,
                      "filters": self.filters})
      return config

  def call(self, x):
    activation_fn = tf.nn.relu

    shortcut = x
    x1 = self.atrous_a1(x)
    x1 = self.norm_a1(x1)
    x1 = activation_fn(x1)
    x1 = self.atrous_a2(x1)
    x1 = self.norm_a2(x1)

    x2 = self.atrous_b1(x)
    x2 = self.norm_b1(x2)
    x2 = activation_fn(x2)
    x2 = self.atrous_b2(x2)
    x2 = self.norm_b2(x2)

    x = tf.add(shortcut, x1)
    x = tf.add(x, x2)
    x = activation_fn(x)
    return x

