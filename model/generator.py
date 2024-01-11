import tensorflow as tf
import tensorflow_addons as tfa

from model.encoder import Encoder
from model.rct import Rct
from model.decoder import Decoder

class Generator(tf.keras.Model):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = Encoder(cfg.weight_decay)
        self.decoder = Decoder(cfg.weight_decay)
        self.rct = Rct(cfg.weight_decay, cfg.batch_size_per_gpu)

    def call(self, images):
        x, sc_0, sc_1, sc_2, sc_3, sc_4 = self.encoder(images)
        x = self.rct(x)
        recon = self.decoder(x, sc_0, sc_1, sc_2, sc_3, sc_4)
        return tf.nn.tanh(recon)
