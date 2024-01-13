import tensorflow as tf
import tensorflow_addons as tfa

from model.encoder import Encoder
from model.rct import Rct
from model.decoder import Decoder

class Generator(tf.keras.Model):
    def __init__(self, decay, batch_size_per_gpu, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decay = decay
        self.batch_size_per_gpu = batch_size_per_gpu

        self.encoder = Encoder(decay)
        self.decoder = Decoder(decay)
        self.rct = Rct(decay, batch_size_per_gpu)

    def get_config(self):
        config = super().get_config()
        config.update({"decay": self.decay, "batch_size_per_gpu": self.batch_size_per_gpu})
        return config

    def call(self, images):
        x, sc_0, sc_1, sc_2, sc_3, sc_4 = self.encoder(images)
        x = self.rct(x)
        recon = self.decoder(x, sc_0, sc_1, sc_2, sc_3, sc_4)
        return tf.nn.tanh(recon)
