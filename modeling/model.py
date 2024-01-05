import tensorflow as tf
import tensorflow_addons as tfa

from modeling.grb import Grb
from modeling.identity import IdentityBlock
from modeling.convolution import ConvolutionalBlock
from modeling.rct import Rct
from modeling.shc import Shc


import modeling.relu as mr

class Model(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grb = Grb(cfg.weight_decay)
        self.identity_block = IdentityBlock(cfg.weight_decay)
        self.convolutional_block = ConvolutionalBlock(cfg.weight_decay)
        self.rct = Rct(cfg.weight_decay, self.cfg.batch_size_per_gpu)
        self.shc = Shc(cfg.weight_decay)

        self.initializer = tf.compat.v1.keras.initializers.glorot_normal()
        self.regularizer = tf.keras.regularizers.L2(self.cfg.weight_decay)

    # def build_regularizer(self):
    #     return tf.keras.regularizers.L2(self.cfg.weight_decay)

    def build_normalizer(self):
        return tfa.layers.InstanceNormalization()

    def build_conv0(self):
        return tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4),
                strides=(2, 2), name='conv0',
                kernel_regularizer=self.regularizer,
                padding='same', kernel_initializer=self.initializer, use_bias=False)

    def build_conv1(self):
        return tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4),
                strides=(2, 2), name='conv1', padding='same',
                kernel_regularizer=self.regularizer,
                kernel_initializer=self.initializer, use_bias=False)

    def build_convT4(self):
        return tf.keras.layers.Conv2DTranspose(512, 4, strides=(2,2),
                activation=None, padding='same', kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                bias_initializer=None, use_bias=False)

    def build_convT3(self):
        return tf.keras.layers.Conv2DTranspose(256, 4, strides=(2,2),
                activation=None, padding='same', kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                bias_initializer=None, use_bias=False)

    def build_convT2(self):
        return tf.keras.layers.Conv2DTranspose(128, 4, strides=(2,2),
                activation=None, padding='same', kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                bias_initializer=None, use_bias=False)

    def build_convT1(self):
        return tf.keras.layers.Conv2DTranspose(64, 4, strides=(2,2),
                activation=None, padding='same', kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                bias_initializer=None, use_bias=False)

    def build_convT0(self):
        return tf.keras.layers.Conv2DTranspose(3, 4, strides=(2,2),
                activation=None, padding='same', kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                bias_initializer=None, use_bias=False)

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, images, reuse=None):
        with tf.compat.v1.variable_scope('GEN', reuse=reuse):
            x = images

            # stage 1

            x = tf.compat.v1.keras.utils.get_or_create_layer("main_conv0", self.build_conv0)(x)
            x = mr.in_lrelu(x, "main_act0")
            short_cut0 = x
            x = tf.compat.v1.keras.utils.get_or_create_layer("main_conv1", self.build_conv1)(x)
            x = mr.in_lrelu(x, "main_act1")
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


            train = tf.compat.v1.keras.utils.get_or_create_layer("main_convT4", self.build_convT4)(train)
            train = tf.compat.v1.keras.utils.get_or_create_layer("main_convT4_in", self.build_normalizer)(train)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut3, sc], axis=3)
            merge = self.shc(merge, short_cut3, 512)
            merge = mr.in_relu(merge, "main_actT4_merge")
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



            train = tf.compat.v1.keras.utils.get_or_create_layer("main_convT3", self.build_convT3)(train)
            train = tf.compat.v1.keras.utils.get_or_create_layer("main_convT3_in", self.build_normalizer)(train)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut2, sc], axis=3)
            merge = self.shc(merge, short_cut2, 256)
            merge = mr.in_relu(merge, "main_actT3_merge")
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

            train = tf.compat.v1.keras.utils.get_or_create_layer("main_convT2", self.build_convT2)(train)
            train = tf.compat.v1.keras.utils.get_or_create_layer("main_convT2_in", self.build_normalizer)(train)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut1, sc], axis=3)
            merge = self.shc(merge, short_cut1, 128)
            merge = mr.in_relu(merge, "main_actT2_merge")
            train = tf.concat(
                [merge, kp], axis=2)

            # stage -1

            train = tf.compat.v1.keras.utils.get_or_create_layer("main_convT1", self.build_convT1)(train)
            train = tf.compat.v1.keras.utils.get_or_create_layer("main_convT1_in", self.build_normalizer)(train)
            sc, kp = tf.split(train, 2, axis=2)
            sc = tf.nn.relu(sc)
            merge = tf.concat([short_cut0, sc], axis=3)
            merge = self.shc(merge, short_cut0, 64)
            merge = mr.in_relu(merge, "main_actT1_merge")
            train = tf.concat(
                [merge, kp], axis=2)

            # stage -0
            recon = tf.compat.v1.keras.utils.get_or_create_layer("main_convT0", self.build_convT0)(train)

        return recon, tf.nn.tanh(recon)


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
    def build_adversarial_global(self, img, reuse=None, name='global'):
        bs = img.get_shape().as_list()[0]
        with tf.compat.v1.variable_scope(name, reuse=reuse):

            def lrelu(x, leak=0.2, name="lrelu"):
                with tf.compat.v1.variable_scope(name):
                    f1 = 0.5 * (1 + leak)
                    f2 = 0.5 * (1 - leak)
                    return f1 * x + f2 * abs(x)

            self.size = 128
            self.activation_fn = lrelu

            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_conv1_" + name, self.build_adversarial_global_conv1)(img)

            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_conv2_" + name, self.build_adversarial_global_conv2)(img)
            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_norm2_" + name, self.build_normalizer)(img)

            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_conv3_" + name, self.build_adversarial_global_conv3)(img)
            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_norm3_" + name, self.build_normalizer)(img)

            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_conv4_" + name, self.build_adversarial_global_conv4)(img)
            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_norm4_" + name, self.build_normalizer)(img)

            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_conv5_" + name, self.build_adversarial_global_conv5)(img)
            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_global_norm5_" + name, self.build_normalizer)(img)

            logit = tf.compat.v1.layers.dense(tf.reshape(
                img, [bs, -1]), 1, activation=None)

        return logit

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
    def build_adversarial_local(self, img, reuse=None, name=None):
        bs = img.get_shape().as_list()[0]
        with tf.compat.v1.variable_scope(name, reuse=reuse):

            def lrelu(x, leak=0.2, name="lrelu"):
                with tf.compat.v1.variable_scope(name):
                    f1 = 0.5 * (1 + leak)
                    f2 = 0.5 * (1 - leak)
                    return f1 * x + f2 * abs(x)

            self.size = 128
            self.activation_fn = lrelu

            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_conv1_" + name, self.build_adversarial_local_conv1)(img)

            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_conv2_" + name, self.build_adversarial_local_conv2)(img)
            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_norm2_" + name, self.build_normalizer)(img)

            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_conv3_" + name, self.build_adversarial_local_conv3)(img)
            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_norm3_" + name, self.build_normalizer)(img)

            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_conv4_" + name, self.build_adversarial_local_conv4)(img)
            img = tf.compat.v1.keras.utils.get_or_create_layer("adversarial_local_norm4_" + name, self.build_normalizer)(img)

            logit = tf.compat.v1.layers.dense(tf.reshape(
                img, [bs, -1]), 1, activation=None)

        return logit


