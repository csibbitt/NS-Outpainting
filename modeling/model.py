import tensorflow as tf
import tensorflow_addons as tfa

from modeling.grb import Grb
from modeling.identity import IdentityBlock
from modeling.convolution import ConvolutionalBlock
from modeling.rct import Rct


import modeling.relu as mr

class Model():
    def __init__(self, cfg):
        self.cfg = cfg
        self.grb = Grb(cfg.weight_decay)
        self.identity_block = IdentityBlock(cfg.weight_decay)
        self.convolutional_block = ConvolutionalBlock(cfg.weight_decay)
        self.rct = Rct(cfg.weight_decay, self.cfg.batch_size_per_gpu)

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
            x = mr.in_lrelu(x)
            short_cut0 = x
            x = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(
                2, 2), name='conv1', padding='same', kernel_regularizer=regularizer, kernel_initializer=initializer, use_bias=False)(x)
            x = mr.in_lrelu(x)
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
            merge = mr.in_relu(merge)
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
            merge = mr.in_relu(merge)
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
            merge = mr.in_relu(merge)
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
            merge = mr.in_relu(merge)
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


