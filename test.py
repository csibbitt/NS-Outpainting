#!/usr/bin/env python
import tensorflow as tf

import numpy as np
import tf_slim as slim
import sys

from PIL import Image

from model.generator import Generator
from model.discriminator import DiscriminatorLocal, DiscriminatorGlobal

from dataset.build_dataset import input_hasher

from contextlib import contextmanager

from model.loss import Loss

# @contextmanager
# def assert_no_variable_creations():
#   """Assert no variables are created in this context manager scope."""
#   def invalid_variable_creator(next_creator, **kwargs):
#     raise ValueError("Attempted to create a new variable instead of reusing an existing one. Args: {}".format(kwargs))

#   with tf.variable_creator_scope(invalid_variable_creator):
#     yield

# @contextmanager
# def catch_and_raise_created_variables():
#   """Raise all variables created within this context manager scope (if any)."""
#   created_vars = []
#   def variable_catcher(next_creator, **kwargs):
#     var = next_creator(**kwargs)
#     created_vars.append(var)
#     return var

#   with tf.variable_creator_scope(variable_catcher):
#     yield
#   if created_vars:
#     raise ValueError("Created vars:", created_vars)

# tf.keras.utils.set_random_seed(1)
# tf.config.experimental.enable_op_determinism()
# tf.random.set_seed(1)

# # random_tool = v1.keras.utils.DeterministicRandomTestTool()
# # with random_tool.scope():


# learning_rate = tf.Variable(0.0001, dtype=tf.float16, shape=[])
# lambda_rec =  tf.Variable(0.998, dtype=tf.float16, shape=[])

# G_opt = tf.keras.optimizers.Adam(
#     learning_rate=learning_rate, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
# D_opt = tf.keras.optimizers.Adam(
#     learning_rate=learning_rate, beta_1=0.5, beta_2=0.9, epsilon=1e-08)

# inputs = tf.zeros( (2, 128, 256, 3))
# groundtruth = inputs

# cfg = type('cfg', (), {'weight_decay': 0.00002, 'batch_size_per_gpu': 1})()
# G = Generator(cfg)
# generator = G
# D_l = DiscriminatorLocal()
# D_g = DiscriminatorGlobal()
# loss = Loss(cfg)

# step = tf.Variable(0, dtype=tf.int64, trainable=False)
# ckpt_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)

# ckpt = tf.train.Checkpoint( step = step,
#                             epoch=ckpt_epoch,
#                             G_opt=G_opt,
#                             D_opt=D_opt,
#                             generator=generator,
#                             discrim_g=loss.discrim_g,
#                             discrim_l=loss.discrim_l)
# status = ckpt.restore('./badcheckpoint/ckpt-25').assert_consumed()

# tf.debugging.enable_check_numerics()

# with tf.GradientTape() as g_G, tf.GradientTape() as g_D:
#     g_G.watch(groundtruth)
#     g_D.watch(groundtruth)

#     left_gt = tf.slice(groundtruth, [0, 0, 0, 0], [2, 128, 128, 3])
#     reconstruction = generator(left_gt)

#     loss_rec = loss.masked_reconstruction_loss(inputs, reconstruction)  #** Could skip this when only training D()
#     loss_adv_G, loss_adv_D = loss.global_and_local_adv_loss(groundtruth, reconstruction) #** Could skip this during G() warmu

#     loss_G = loss_adv_G * (1 - lambda_rec) + loss_rec * lambda_rec + tf.reduce_sum(generator.losses)
#     loss_D = loss_adv_D

#     var_G = generator.trainable_variables
#     var_D = loss.discrim_l.trainable_variables + loss.discrim_g.trainable_variables

#     grad_g = G_opt.compute_gradients(loss_G, var_G, tape=g_G)
#     grad_d = D_opt.compute_gradients(loss_D, var_D, tape=g_D)

#     G_opt.apply_gradients(grad_g)
#     D_opt.apply_gradients(grad_d)

# print(loss_G.numpy())

# recon = G(inputs)

# print(len(G.trainable_variables) + len(D_l.trainable_variables) + len(D_g.trainable_variables))

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))


# reader = tf.train.load_checkpoint('./badcheckpoint/ckpt-25')



# reader.get_tensor('step/.ATTRIBUTES/VARIABLE_VALUE')

#tf.math.top_k( list(tf.train.list_variables('./badcheckpoint/ckpt-25')), k=5, sorted=True )

# for var in tf.train.list_variables('./badcheckpoint/ckpt-25'):
#   print(f'{var} : {var.numpy()}')



# for var in G.trainable_variables:
#     print(var)

# def parse_trainset(example_proto):

#     dics = {}
#     dics['image'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

#     parsed_example = tf.io.parse_single_example(
#         serialized=example_proto, features=dics)
#     image = tf.io.decode_raw(parsed_example['image'], out_type=tf.uint8)

#     #image = tf.reshape(image, shape=[72 * 2, 216 * 2, 3])
#     image = tf.reshape(image, shape=[64 * 2, 128 * 2, 3])

#     # image = tf.image.random_crop(image, [64 * 2, 128 * 2, 3])
#     # image = tf.image.random_flip_left_right(image)
#     image = tf.cast(image, tf.float16) / 255.
#     image = 2. * image - 1.

#     return image

# trainset = tf.data.TFRecordDataset(filenames=['tf_dataset_new/testset.tfr'])
# trainset = trainset.map(parse_trainset, num_parallel_calls=tf.data.AUTOTUNE)
# trainset = trainset.batch(1)
# train_im = iter(trainset)

# for image in train_im:
#   hash = input_hasher(image)
#   Image.fromarray((255. * (image[0].numpy() + 1) / 2.).astype(np.uint8)).save(f"dataset/testset/{hash}.jpg")