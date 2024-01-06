import tensorflow as tf
import tensorflow_addons as tfa

from model.rct import Rct
from model.encoder import Encoder
from model.decoder import Decoder

import model.relu as mr

class Generator(tf.keras.Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.encoder = Encoder(cfg.weight_decay)
        self.decoder = Decoder(cfg.weight_decay)

        self.rct = Rct(cfg.weight_decay, self.cfg.batch_size_per_gpu)
        self.initializer = tf.compat.v1.keras.initializers.glorot_normal()
        self.regularizer = tf.keras.regularizers.L2(cfg.weight_decay)

    def call(self, images, reuse=None):
        with tf.compat.v1.variable_scope('GEN', reuse=reuse):
            x, shortcuts = self.encoder(images)
            x = self.rct(x)
            recon = self.decoder(x, shortcuts)
        return recon, tf.nn.tanh(recon)

    def build_normalizer(self):
        return tfa.layers.InstanceNormalization()

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


