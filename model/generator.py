import tensorflow as tf
import tensorflow_addons as tfa

from model.encoder import Encoder
from model.rct import Rct
from model.decoder import Decoder

import model.relu as mr

class Generator(tf.keras.Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        self.encoder = Encoder(cfg.weight_decay)
        self.decoder = Decoder(cfg.weight_decay)
        self.rct = Rct(cfg.weight_decay, self.cfg.batch_size_per_gpu)

    def call(self, images, reuse=None):
        with tf.compat.v1.variable_scope('GEN', reuse=reuse):
            x, shortcuts = self.encoder(images)
            x = self.rct(x)
            recon = self.decoder(x, shortcuts)
        return recon, tf.nn.tanh(recon)
