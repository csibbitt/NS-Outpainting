import tensorflow as tf
import tensorflow_addons as tfa


def build_normalizer():
  return tfa.layers.InstanceNormalization()

class DiscriminatorGlobal(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def build_adversarial_global_conv1(self):
    return tf.keras.layers.Conv2D(filters=self.size / 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same')

  def build_adversarial_global_conv2(self):
    return tf.keras.layers.Conv2D(filters=self.size, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)

  def build_adversarial_global_conv3(self):
    return tf.keras.layers.Conv2D(filters=self.size * 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)

  def build_adversarial_global_conv4(self):
    return tf.keras.layers.Conv2D(filters=self.size * 4, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)

  def build_adversarial_global_conv5(self):
    return tf.keras.layers.Conv2D(filters=self.size * 4, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, img, name='DIS'):
    bs = img.get_shape().as_list()[0]

    def lrelu(x, leak=0.2, name="lrelu"):
      with tf.compat.v1.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    self.size = 128
    self.activation_fn = lrelu

    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_conv1_" + name, self.build_adversarial_global_conv1)(img)

    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_conv2_" + name, self.build_adversarial_global_conv2)(img)
    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_norm2_" + name, build_normalizer)(img)

    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_conv3_" + name, self.build_adversarial_global_conv3)(img)
    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_norm3_" + name, build_normalizer)(img)

    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_conv4_" + name, self.build_adversarial_global_conv4)(img)
    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_norm4_" + name, build_normalizer)(img)

    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_conv5_" + name, self.build_adversarial_global_conv5)(img)
    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_norm5_" + name, build_normalizer)(img)

    logit = tf.compat.v1.layers.dense(tf.reshape(
        img, [bs, -1]), 1, activation=None, name=name+"/dense")

    return logit


class DiscriminatorLocal(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def build_adversarial_local_conv1(self):
      return tf.keras.layers.Conv2D(filters=self.size / 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same')

  def build_adversarial_local_conv2(self):
      return tf.keras.layers.Conv2D(filters=self.size, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)

  def build_adversarial_local_conv3(self):
      return tf.keras.layers.Conv2D(filters=self.size * 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)

  def build_adversarial_local_conv4(self):
      return tf.keras.layers.Conv2D(filters=self.size * 2, kernel_size=4, kernel_initializer=None,
                          strides=(2,2), activation=self.activation_fn, padding='same', use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, img, name='DIS2'):

    bs = img.get_shape().as_list()[0]

    def lrelu(x, leak=0.2, name="lrelu"):
      with tf.compat.v1.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    self.size = 128
    self.activation_fn = lrelu

    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_conv1_" + name, self.build_adversarial_local_conv1)(img)

    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_conv2_" + name, self.build_adversarial_local_conv2)(img)
    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_norm2_" + name, build_normalizer)(img)

    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_conv3_" + name, self.build_adversarial_local_conv3)(img)
    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_norm3_" + name, build_normalizer)(img)

    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_conv4_" + name, self.build_adversarial_local_conv4)(img)
    img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_norm4_" + name, build_normalizer)(img)

    logit = tf.compat.v1.layers.dense(tf.reshape(
        img, [bs, -1]), 1, activation=None, name=name+"/dense")

    return logit


