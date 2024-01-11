import tensorflow as tf
import tensorflow_addons as tfa


class DiscriminatorGlobal(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    size = 128
    activation_fn = tf.nn.leaky_relu

    self.conv_1 = tf.keras.layers.Conv2D(filters=size / 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=activation_fn, padding='same')
    
    self.conv_2 = tf.keras.layers.Conv2D(filters=size, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=activation_fn, padding='same', use_bias=False)
    self.norm_2 = tfa.layers.InstanceNormalization()

    self.conv_3 = tf.keras.layers.Conv2D(filters=size * 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=activation_fn, padding='same', use_bias=False)
    self.norm_3 = tfa.layers.InstanceNormalization()

    self.conv_4 = tf.keras.layers.Conv2D(filters=size * 4, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=activation_fn, padding='same', use_bias=False)
    self.norm_4 = tfa.layers.InstanceNormalization()

    self.conv_5 = tf.keras.layers.Conv2D(filters=size * 4, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=activation_fn, padding='same', use_bias=False)
    self.norm_5 = tfa.layers.InstanceNormalization()

    self.dense =  tf.keras.layers.Dense(1, activation=None, kernel_initializer=None)

  def call(self, img):
    bs = img.get_shape().as_list()[0]

    img = self.conv_1(img)

    img = self.conv_2(img)
    img = self.norm_2(img)

    img = self.conv_3(img)
    img = self.norm_3(img)

    img = self.conv_4(img)
    img = self.norm_4(img)

    img = self.conv_5(img)
    img = self.norm_5(img)

    logit = self.dense(tf.reshape(img, [bs, -1]))

    return logit


class DiscriminatorLocal(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    size = 128
    activation_fn = tf.nn.leaky_relu

    self.conv_1 = tf.keras.layers.Conv2D(filters=size / 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=activation_fn, padding='same')
    
    self.conv_2 = tf.keras.layers.Conv2D(filters=size, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=activation_fn, padding='same', use_bias=False)
    self.norm_2 = tfa.layers.InstanceNormalization()

    self.conv_3 = tf.keras.layers.Conv2D(filters=size * 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=activation_fn, padding='same', use_bias=False)
    self.norm_3 = tfa.layers.InstanceNormalization()

    self.conv_4 = tf.keras.layers.Conv2D(filters=size * 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=activation_fn, padding='same', use_bias=False)
    self.norm_4 = tfa.layers.InstanceNormalization()

    self.dense = tf.keras.layers.Dense(1, activation=None, kernel_initializer=None)

  def call(self, img):
    bs = img.get_shape().as_list()[0]

    img = self.conv_1(img)

    img = self.conv_2(img)
    img = self.norm_2(img)
    
    img = self.conv_3(img)
    img = self.norm_3(img)
    
    img = self.conv_4(img)
    img = self.norm_4(img)

    logit = self.dense(tf.reshape(img, [bs, -1]))

    return logit


