#!/usr/bin/env python
import argparse



parser = argparse.ArgumentParser(description='Convert a checkpoint to saved generator model')
parser.add_argument('filename')
parser.add_argument('-o', '--output-dir', default='saved-model.tf', help='defaults to "saved-model.tf"',)

args = parser.parse_args()

import tensorflow as tf
from model.generator import Generator

generator = Generator(name='model', decay=0.00002, batch_size_per_gpu=1)
generator(tf.ones([1,128,128,3]))

print('Loading checkpoint')
ckpt = tf.train.Checkpoint(generator=generator)
ckpt.restore(args.filename).assert_existing_objects_matched().expect_partial()

print('Saving Model')
generator.save(args.output_dir)
