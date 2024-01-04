import tensorflow as tf

import modeling.relu as mr

# Global residual block
class Rct(tf.keras.layers.Layer):

  def build_regularizer(self):
    return tf.keras.regularizers.L2(self.decay)

  def __init__(self, decay, batch_size_per_gpu, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.decay = decay
    self.batch_size_per_gpu = batch_size_per_gpu

  def build_conv1(self):
    return tf.keras.layers.Conv2D(self.size, 1, strides=(1,1), activation=None,
                    padding='same',
                    kernel_regularizer=tf.compat.v1.keras.utils.get_or_create_layer("rct_regularizer1", self.build_regularizer),
                    bias_initializer=None, use_bias=False)

  def build_conv2(self):
    return tf.keras.layers.Conv2D(self.output_size, 1, strides=(1,1), activation=None,
                    padding='same',
                    kernel_regularizer=tf.compat.v1.keras.utils.get_or_create_layer("rct_regularizer2", self.build_regularizer),
                    bias_initializer=None, use_bias=False)

  @tf.compat.v1.keras.utils.track_tf1_style_variables
  def call(self, x):
    self.output_size = x.get_shape().as_list()[3]
    self.size = 512
    layer_num = 2
    activation_fn = tf.tanh
    x = tf.compat.v1.keras.utils.get_or_create_layer("rct_conv1", self.build_conv1)(x)
    x = mr.in_lrelu(x, "conv1_act")
    x = tf.transpose(a=x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, [-1, 4, 4 * self.size])
    x = tf.transpose(a=x, perm=[1, 0, 2])
    # encoder_inputs = x
    x = tf.reshape(x, [-1, 4 * self.size])
    x_split = tf.split(x, 4, 0)

    ys = []
    with tf.compat.v1.variable_scope('LSTM'):
        with tf.compat.v1.variable_scope('encoder'):
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
                4 * self.size, activation=activation_fn)
            lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell( # WARNING:tensorflow:At least two cells provided to MultiRNNCell are the same object and will share weights.
                [lstm_cell] * layer_num, state_is_tuple=True)

        init_state = lstm_cell.zero_state(self.batch_size_per_gpu, dtype=tf.float32)
        now, _state = lstm_cell(x_split[0], init_state)
        now, _state = lstm_cell(x_split[1], _state)
        now, _state = lstm_cell(x_split[2], _state)
        now, _state = lstm_cell(x_split[3], _state)

        with tf.compat.v1.variable_scope('decoder'):
            lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
                4 * self.size, activation=activation_fn)
            lstm_cell2 = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [lstm_cell] * layer_num, state_is_tuple=True)
        #predict
        now, _state = lstm_cell2(x_split[3], _state)
        ys.append(tf.reshape(now, [-1, 4, 1, self.size]))
        now, _state = lstm_cell2(now, _state)
        ys.append(tf.reshape(now, [-1, 4, 1, self.size]))
        now, _state = lstm_cell2(now, _state)
        ys.append(tf.reshape(now, [-1, 4, 1, self.size]))
        now, _state = lstm_cell2(now, _state)
        ys.append(tf.reshape(now, [-1, 4, 1, self.size]))


    y = tf.concat(ys, axis=2)

    y = tf.compat.v1.keras.utils.get_or_create_layer("rct_conv2", self.build_conv2)(y)
    y = mr.in_lrelu(y, "conv2_act")
    return y