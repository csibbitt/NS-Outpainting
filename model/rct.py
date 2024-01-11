import tensorflow as tf
import tensorflow_addons as tfa

# Recurrent Content Transfer
class Rct(tf.keras.layers.Layer):

  def __init__(self, decay, batch_size_per_gpu, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.output_size = 1024 #** Orig was: x.get_shape().as_list()[3]
    self.size = 512
    self.decay = decay
    self.batch_size_per_gpu = batch_size_per_gpu

    self.regularizer = tf.keras.regularizers.L2(self.decay)

    encoder_lstm_cell = tf.keras.layers.LSTMCell(4 * self.size, recurrent_activation=tf.tanh, kernel_initializer=None, recurrent_initializer=None)
    self.encoder_lstm = tf.keras.layers.StackedRNNCells([encoder_lstm_cell] * 2)   #** Is this a bug in the original code, or do we mean to share weights in both cells?

    decoder_lstm_cell = tf.keras.layers.LSTMCell(4 * self.size, recurrent_activation=tf.tanh, kernel_initializer=None, recurrent_initializer=None)
    self.decoder_lstm = tf.keras.layers.StackedRNNCells([decoder_lstm_cell] * 2)   #** Is this a bug in the original code, or do we mean to share weights in both cells?

    self.conv_1 = tf.keras.layers.Conv2D(self.size, 1, strides=(1,1), activation=None,
                    padding='same',
                    kernel_regularizer=self.regularizer, kernel_initializer=None,
                    bias_initializer=None, use_bias=False)
    self.norm_1 = tfa.layers.InstanceNormalization()

    self.conv_2 = tf.keras.layers.Conv2D(self.output_size, 1, strides=(1,1), activation=None,
                    padding='same',
                    kernel_regularizer=self.regularizer, kernel_initializer=None,
                    bias_initializer=None, use_bias=False)
    self.norm_2 = tfa.layers.InstanceNormalization()

  def call(self, x):

    x = self.conv_1(x)
    x = self.norm_1(x)
    x = tf.nn.leaky_relu(x)
    x = tf.transpose(a=x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, [-1, 4, 4 * self.size])
    x = tf.transpose(a=x, perm=[1, 0, 2])
    # encoder_inputs = x
    x = tf.reshape(x, [-1, 4 * self.size])
    x_split = tf.split(x, 4, 0)

    init_state =  self.encoder_lstm.get_initial_state(x_split[0], batch_size=self.batch_size_per_gpu, dtype=tf.float32)
    now, _state = self.encoder_lstm(x_split[0], init_state)
    now, _state = self.encoder_lstm(x_split[1], _state)
    now, _state = self.encoder_lstm(x_split[2], _state)
    now, _state = self.encoder_lstm(x_split[3], _state)

    #predict
    now, _state = self.decoder_lstm(x_split[3], _state)
    y1 = tf.reshape(now, [-1, 4, 1, self.size])
    now, _state = self.decoder_lstm(now, _state)
    y2 = tf.reshape(now, [-1, 4, 1, self.size])
    now, _state = self.decoder_lstm(now, _state)
    y3 = tf.reshape(now, [-1, 4, 1, self.size])
    now, _state = self.decoder_lstm(now, _state)
    y4 = tf.reshape(now, [-1, 4, 1, self.size])

    y = tf.concat([y1, y2, y3, y4], axis=2)

    y = self.conv_2(y)
    y = self.norm_2(y)
    y =  tf.nn.leaky_relu(y)
    return y