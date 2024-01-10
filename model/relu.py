
import tensorflow as tf
import tensorflow_addons as tfa

def build_normalizer():
  return tfa.layers.InstanceNormalization()

def leaky_relu(x, name="", leak=0.2):
  f1 = 0.5 * (1 + leak)
  f2 = 0.5 * (1 - leak)
  return f1 * x + f2 * abs(x)

@tf.compat.v1.keras.utils.track_tf1_style_variables
def in_lrelu(x, name=""):
  x = tf.compat.v1.keras.utils.get_or_create_layer("in_lrelu_nomalizer_" + name, build_normalizer)(x)
  x = leaky_relu(x)
  return x

@tf.compat.v1.keras.utils.track_tf1_style_variables
def in_relu(x, name=""):
  x = tf.compat.v1.keras.utils.get_or_create_layer("in_relu_nomalizer_" + name, build_normalizer)(x)
  x = tf.nn.relu(x)
  return x