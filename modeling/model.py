import tensorflow as tf
import tensorflow_addons as tfa

from modeling.grb import Grb

class Model():
    def __init__(self, cfg):
        self.cfg = cfg
        self.grb = Grb(cfg.weight_decay)

    def identity_block(self, X_input, kernel_size, filters, stage, block, is_relu=False):
        if is_relu:
            activation_fn=tf.nn.relu
        else:
            activation_fn=self.leaky_relu

        normalizer_fn = tfa.layers.InstanceNormalization


        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'

        with tf.compat.v1.variable_scope("id_block_stage" + str(stage) + block):
            filter1, filter2, filter3 = filters
            X_shortcut = X_input
            regularizer = tf.keras.regularizers.L2(self.cfg.weight_decay)
            initializer = tf.compat.v1.keras.initializers.glorot_normal()

            # First component of main path
            x = tf.keras.layers.Conv2D(filter1,
                                 kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2a', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)(X_input)
            x = normalizer_fn()(x)
            x = activation_fn(x)

            # Second component of main path
            x = tf.keras.layers.Conv2D(filter2, (kernel_size, kernel_size),
                                 padding='same', name=conv_name_base + '2b', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)(x)
            x = normalizer_fn()(x)
            x = activation_fn(x)

            # Third component of main path
            x = tf.keras.layers.Conv2D(filter3, kernel_size=(
                1, 1), name=conv_name_base + '2c', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)(x)
            x = normalizer_fn()(x)

            # Final step: Add shortcut value to main path, and pass it through
            x = tf.add(x, X_shortcut)
            x = activation_fn(x)

        return x

    def convolutional_block(self, X_input, kernel_size, filters, stage, block, stride=2, is_relu=False):
        
        if is_relu:
            activation_fn=tf.nn.relu
            
        else:
            activation_fn=self.leaky_relu

        normalizer_fn = tfa.layers.InstanceNormalization

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'

        with tf.compat.v1.variable_scope("conv_block_stage" + str(stage) + block):

            regularizer = tf.keras.regularizers.L2(self.cfg.weight_decay)
            initializer = tf.compat.v1.keras.initializers.glorot_normal()
            # initializer = tf.variance_scaling_initializer(scale=1.0,mode='fan_in')

            # Retrieve Filters
            filter1, filter2, filter3 = filters

            # Save the input value
            X_shortcut = X_input

            # First component of main path
            x = tf.keras.layers.Conv2D(filter1,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 name=conv_name_base + '2a', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)(X_input)
            x = normalizer_fn()(x)
            x = activation_fn(x)

            # Second component of main path
            x = tf.keras.layers.Conv2D(filter2, (kernel_size, kernel_size), strides=(stride, stride), name=conv_name_base +
                                 '2b', padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)(x)
            x = normalizer_fn()(x)
            x = activation_fn(x)

            # Third component of main path
            x = tf.keras.layers.Conv2D(filter3, (1, 1), name=conv_name_base + '2c',
                                 kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)(x)
            x = normalizer_fn()(x)


            # SHORTCUT PATH
            X_shortcut = tf.keras.layers.Conv2D(filter3, (1, 1),
                                          strides=(stride, stride), name=conv_name_base + '1', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)(X_shortcut)
            X_shortcut = normalizer_fn()(X_shortcut)

            # Final step: Add shortcut value to main path, and pass it through
            # a RELU activation
            x = tf.add(X_shortcut, x)
            x = activation_fn(x)

        return x

    def leaky_relu(self, x, name=None, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    def in_lrelu(self, x, name=None):
        x = tfa.layers.InstanceNormalization()(x)
        x = self.leaky_relu(x)
        return x

    def in_relu(self, x, name=None):
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.nn.relu(x)
        return x

    def rct(self, x):
        regularizer = tf.keras.regularizers.L2(self.cfg.weight_decay)
        output_size = x.get_shape().as_list()[3]
        size = 512
        layer_num = 2
        activation_fn = tf.tanh
        x = tf.keras.layers.Conv2D(size, 1, strides=(1,1), activation=None,
                       padding='SAME', kernel_regularizer=regularizer, bias_initializer=None)(x)
        x = self.in_lrelu(x)
        x = tf.transpose(a=x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [-1, 4, 4 * size])
        x = tf.transpose(a=x, perm=[1, 0, 2])
        # encoder_inputs = x
        x = tf.reshape(x, [-1, 4 * size])
        x_split = tf.split(x, 4, 0)

        ys = []
        with tf.compat.v1.variable_scope('LSTM'):
            with tf.compat.v1.variable_scope('encoder'):
                lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
                    4 * size, activation=activation_fn)
                lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell( # WARNING:tensorflow:At least two cells provided to MultiRNNCell are the same object and will share weights.
                    [lstm_cell] * layer_num, state_is_tuple=True)
            
            init_state = lstm_cell.zero_state(self.cfg.batch_size_per_gpu, dtype=tf.float32)
            now, _state = lstm_cell(x_split[0], init_state)
            now, _state = lstm_cell(x_split[1], _state)
            now, _state = lstm_cell(x_split[2], _state)
            now, _state = lstm_cell(x_split[3], _state)

            with tf.compat.v1.variable_scope('decoder'):
                lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
                    4 * size, activation=activation_fn)
                lstm_cell2 = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                    [lstm_cell] * layer_num, state_is_tuple=True)
            #predict
            now, _state = lstm_cell2(x_split[3], _state)
            ys.append(tf.reshape(now, [-1, 4, 1, size]))
            now, _state = lstm_cell2(now, _state)
            ys.append(tf.reshape(now, [-1, 4, 1, size]))
            now, _state = lstm_cell2(now, _state)
            ys.append(tf.reshape(now, [-1, 4, 1, size]))
            now, _state = lstm_cell2(now, _state)
            ys.append(tf.reshape(now, [-1, 4, 1, size]))
        

        y = tf.concat(ys, axis=2)

        y = tf.keras.layers.Conv2D(output_size, 1, strides=(1,1), activation=None,
                       padding='SAME', kernel_regularizer=regularizer, bias_initializer=None)(y)
        y = self.in_lrelu(y)
        return y



    def shc(self, x, shortcut, channels):
        regularizer = tf.keras.regularizers.L2(self.cfg.weight_decay)
        x = tf.keras.layers.Conv2D(channels / 2, 1, strides=(1,1), activation=tf.nn.relu,
                      padding='SAME', kernel_regularizer=regularizer)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.Conv2D(channels / 2, 3, strides=(1,1), activation=tf.nn.relu,
                      padding='SAME', kernel_regularizer=regularizer)(x)
        x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.Conv2D(channels, 1, strides=(1,1), activation=None,
                      padding='SAME', kernel_regularizer=regularizer)(x)
        x = tfa.layers.InstanceNormalization()(x)
        return tf.add(shortcut, x)




    def build_reconstruction(self, images, reuse=None):

        with tf.compat.v1.variable_scope('GEN', reuse=reuse):
            x = images
            normalizer_fn = tfa.layers.InstanceNormalization
            regularizer = tf.keras.regularizers.L2(self.cfg.weight_decay)
            initializer = tf.compat.v1.keras.initializers.glorot_normal()
            # stage 1

            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(
                2, 2), name='conv0', kernel_regularizer=regularizer, padding='same', kernel_initializer=initializer, use_bias=False)(x)
            x = self.in_lrelu(x)
            short_cut0 = x
            x = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(
                2, 2), name='conv1', padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)(x)
            x = self.in_lrelu(x)
            short_cut1 = x

            # stage 2
            x = self.convolutional_block(x, kernel_size=3, filters=[
                                         64, 64, 256], stage=2, block='a', stride=2)
            x = self.identity_block(
                x, 3, [64, 64, 256], stage=2, block='b')
            x = self.identity_block(
                x, 3, [64, 64, 256], stage=2, block='c')
            short_cut2 = x

            # stage 3
            x = self.convolutional_block(x, kernel_size=3, filters=[128, 128, 512],
                                         stage=3, block='a', stride=2)
            x = self.identity_block(
                x, 3, [128, 128, 512], stage=3, block='b')
            x = self.identity_block(
                x, 3, [128, 128, 512], stage=3, block='c')
            x = self.identity_block(
                x, 3, [128, 128, 512], stage=3, block='d',)
            short_cut3 = x

            # stage 4
            x = self.convolutional_block(x, kernel_size=3, filters=[
                                         256, 256, 1024], stage=4, block='a', stride=2)
            x = self.identity_block(
                x, 3, [256, 256, 1024], stage=4, block='b')
            x = self.identity_block(
                x, 3, [256, 256, 1024], stage=4, block='c')
            x = self.identity_block(
                x, 3, [256, 256, 1024], stage=4, block='d')
            x = self.identity_block(
                x, 3, [256, 256, 1024], stage=4, block='e')
            short_cut4 = x
            
            # rct transfer
            train = self.rct(x)


            # stage -4
            train = tf.concat([short_cut4, train], axis=2)

            train = self.grb(train, 1024, 1, 't4')
            train = self.identity_block(
                train, 3, [256, 256, 1024], stage=-4, block='b', is_relu=True)
            train = self.identity_block(
                train, 3, [256, 256, 1024], stage=-4, block='c', is_relu=True)
            

            train = tf.keras.layers.Conv2DTranspose(512, 4, strides=(2,2),
                                        activation=None, padding='SAME', kernel_initializer=initializer, kernel_regularizer=regularizer, bias_initializer=None)(train)
            train = tfa.layers.InstanceNormalization()(train)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut3, sc], axis=3)
            merge = self.shc(merge, short_cut3, 512)
            merge = self.in_relu(merge)
            train = tf.concat(
                [merge, kp], axis=2)


            # stage -3
            train = self.grb(train, 512, 2, 't3')
            train = self.identity_block(
                train, 3, [128, 128, 512], stage=-3, block='b', is_relu=True)
            train = self.identity_block(
                train, 3, [128, 128, 512], stage=-3, block='c', is_relu=True)
            train = self.identity_block(
                train, 3, [128, 128, 512], stage=-3, block='d', is_relu=True)
            
            

            train = tf.keras.layers.Conv2DTranspose(256, 4, strides=(2,2),
                                        activation=None, padding='SAME', kernel_initializer=initializer, kernel_regularizer=regularizer, bias_initializer=None)(train)
            train = tfa.layers.InstanceNormalization()(train)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut2, sc], axis=3)
            merge = self.shc(merge, short_cut2, 256)
            merge = self.in_relu(merge)
            train = tf.concat(
                [merge, kp], axis=2)

            # stage -2
            train = self.grb(train, 256, 4, 't2')
            train = self.identity_block(
                train, 3, [64, 64, 256], stage=-2, block='b', is_relu=True)
            train = self.identity_block(
                train, 3, [64, 64, 256], stage=-2, block='c', is_relu=True)
            train = self.identity_block(
                train, 3, [64, 64, 256], stage=-2, block='d', is_relu=True)
            train = self.identity_block(
                train, 3, [64, 64, 256], stage=-2, block='e', is_relu=True)

            train = tf.keras.layers.Conv2DTranspose(128, 4, strides=(2,2),
                                        activation=None, padding='SAME', kernel_initializer=initializer, kernel_regularizer=regularizer, bias_initializer=None)(train)
            train = tfa.layers.InstanceNormalization()(train)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut1, sc], axis=3)
            merge = self.shc(merge, short_cut1, 128)
            merge = self.in_relu(merge)
            train = tf.concat(
                [merge, kp], axis=2)
 

            # stage -1

            train = tf.keras.layers.Conv2DTranspose(64, 4, strides=(2,2),
                                        activation=None, padding='SAME', kernel_initializer=initializer, kernel_regularizer=regularizer, bias_initializer=None)(train)
            train = tfa.layers.InstanceNormalization()(train)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut0, sc], axis=3)
            merge = self.shc(merge, short_cut0, 64)
            merge = self.in_relu(merge)
            train = tf.concat(
                [merge, kp], axis=2)

            # stage -0
            recon = tf.keras.layers.Conv2DTranspose(3, 4, strides=(2,2),
                                        activation=None, padding='SAME', kernel_initializer=initializer, kernel_regularizer=regularizer, bias_initializer=None)(train)

        return recon, tf.nn.tanh(recon)

    def build_adversarial_global(self, img, reuse=None, name=None):
        bs = img.get_shape().as_list()[0]
        with tf.compat.v1.variable_scope(name, reuse=reuse):

            def lrelu(x, leak=0.2, name="lrelu"):
                with tf.compat.v1.variable_scope(name):
                    f1 = 0.5 * (1 + leak)
                    f2 = 0.5 * (1 - leak)
                    return f1 * x + f2 * abs(x)

            size = 128
            activation_fn = lrelu

            img = tf.keras.layers.Conv2D(filters=size / 2, kernel_size=4,
                            strides=(2,2), activation=activation_fn)(img)
            img = tf.keras.layers.Conv2D(filters=size, kernel_size=4,
                            strides=(2,2), activation=activation_fn)(img)
            img = tfa.layers.InstanceNormalization()(img)
            img = tf.keras.layers.Conv2D(filters=size * 2, kernel_size=4,
                            strides=(2,2), activation=activation_fn)(img)
            img = tfa.layers.InstanceNormalization()(img)
            img = tf.keras.layers.Conv2D(filters=size * 4, kernel_size=4,
                            strides=(2,2), activation=activation_fn)(img)
            img = tfa.layers.InstanceNormalization()(img)
            img = tf.keras.layers.Conv2D(filters=size * 4, kernel_size=4,
                            strides=(2,2), activation=activation_fn)(img)
            img = tfa.layers.InstanceNormalization()(img)

            logit = tf.compat.v1.layers.dense(tf.reshape(
                img, [bs, -1]), 1, activation=None)

        return logit

    def build_adversarial_local(self, img, reuse=None, name=None):
        bs = img.get_shape().as_list()[0]
        with tf.compat.v1.variable_scope(name, reuse=reuse):

            def lrelu(x, leak=0.2, name="lrelu"):
                with tf.compat.v1.variable_scope(name):
                    f1 = 0.5 * (1 + leak)
                    f2 = 0.5 * (1 - leak)
                    return f1 * x + f2 * abs(x)

            size = 128
            activation_fn = lrelu

            img = tf.keras.layers.Conv2D(filters=size / 2, kernel_size=4,
                            strides=(2,2), activation=activation_fn)(img)
            img = tf.keras.layers.Conv2D(filters=size, kernel_size=4,
                            strides=(2,2), activation=activation_fn)(img)
            img = tfa.layers.InstanceNormalization()(img)
            img = tf.keras.layers.Conv2D(filters=size * 2, kernel_size=4,
                            strides=(2,2), activation=activation_fn)(img)
            img = tfa.layers.InstanceNormalization()(img)
            img = tf.keras.layers.Conv2D(filters=size * 2, kernel_size=4,
                            strides=(2,2), activation=activation_fn)(img)
            img = tfa.layers.InstanceNormalization()(img)

            logit = tf.compat.v1.layers.dense(tf.reshape(
                img, [bs, -1]), 1, activation=None)

        return logit


