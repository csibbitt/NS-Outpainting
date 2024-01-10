import tensorflow as tf
import tensorflow_addons as tfa


class DiscriminatorGlobal(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.conv_1 = tf.keras.layers.Conv2D(filters=self.size / 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same')
    
    self.conv_2 = tf.keras.layers.Conv2D(filters=self.size, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)
    self.norm_2 = tfa.layers.InstanceNormalization()

    self.conv_3 = tf.keras.layers.Conv2D(filters=self.size * 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)
    self.norm_3 = tfa.layers.InstanceNormalization()

    self.conv_4 = tf.keras.layers.Conv2D(filters=self.size * 4, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)
    self.norm_4 = tfa.layers.InstanceNormalization()

    self.conv_5 = tf.keras.layers.Conv2D(filters=self.size * 4, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)
    self.norm_5 = tfa.layers.InstanceNormalization()

    self.dense =  tf.keras.layers.Dense(1, activation=None, kernel_initializer=None)

  def call(self, img, name='DIS'):
    bs = img.get_shape().as_list()[0]

    def lrelu(x, leak=0.2):
      f1 = 0.5 * (1 + leak)
      f2 = 0.5 * (1 - leak)
      return f1 * x + f2 * abs(x)

    self.size = 128
    self.activation_fn = lrelu

    img = self.conv_1(img)

    img = self.conv_2(img)
    img = self.norm_2(img)

    img = self.conv_3(img)
    img = self.norm_3(img)

    img = self.conv_4(img)
    img = self.norm_4(img)

    img = self.conv_5(img)
    img = self.norm_5(img)

    logit = self.dense(tf.reshape(img, [bs, -1]), 1, activation=None)

    return logit


class DiscriminatorLocal(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.conv_1 = tf.keras.layers.Conv2D(filters=self.size / 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same')
    
    self.conv_2 = tf.keras.layers.Conv2D(filters=self.size, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)
    self.norm_2 = tfa.layers.InstanceNormalization()

    self.conv_3 = tf.keras.layers.Conv2D(filters=self.size * 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)
    self.norm_3 = tfa.layers.InstanceNormalization()

    self.conv_4 = tf.keras.layers.Conv2D(filters=self.size * 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)
    self.norm_4 = tfa.layers.InstanceNormalization()

    self.dense = tf.keras.layers.Dense(1, activation=None, kernel_initializer=None)

  def call(self, img, name='DIS2'):
    bs = img.get_shape().as_list()[0]

    def lrelu(x, leak=0.2, name="lrelu"):
      f1 = 0.5 * (1 + leak)
      f2 = 0.5 * (1 - leak)
      return f1 * x + f2 * abs(x)

    self.size = 128
    self.activation_fn = lrelu

    img = self.conv_1(img)

    img = self.conv_2(img)
    img = self.norm_2(img)
    
    img = self.conv_3(img)
    img = self.norm_3(img)
    
    img = self.conv_4(img)
    img = self.norm_4(img)

    logit = self.dense(tf.reshape(img, [bs, -1]))

    return logit


