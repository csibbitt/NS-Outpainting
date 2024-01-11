#!/usr/bin/env python
import tensorflow as tf

import numpy as np
import tf_slim as slim
import sys

from model.generator import Generator
from model.discriminator import DiscriminatorLocal, DiscriminatorGlobal


from contextlib import contextmanager

from model.loss import Loss

@contextmanager
def assert_no_variable_creations():
  """Assert no variables are created in this context manager scope."""
  def invalid_variable_creator(next_creator, **kwargs):
    raise ValueError("Attempted to create a new variable instead of reusing an existing one. Args: {}".format(kwargs))

  with tf.variable_creator_scope(invalid_variable_creator):
    yield

@contextmanager
def catch_and_raise_created_variables():
  """Raise all variables created within this context manager scope (if any)."""
  created_vars = []
  def variable_catcher(next_creator, **kwargs):
    var = next_creator(**kwargs)
    created_vars.append(var)
    return var

  with tf.variable_creator_scope(variable_catcher):
    yield
  if created_vars:
    raise ValueError("Created vars:", created_vars)

# tf.keras.utils.set_random_seed(1)
# tf.config.experimental.enable_op_determinism()
# tf.random.set_seed(1)

# # random_tool = v1.keras.utils.DeterministicRandomTestTool()
# # with random_tool.scope():


# learning_rate = tf.Variable(0.0001, dtype=tf.float32, shape=[])
# lambda_rec =  tf.Variable(0.998, dtype=tf.float32, shape=[])

# G_opt = tf.keras.optimizers.Adam(
#     learning_rate=learning_rate, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
# D_opt = tf.keras.optimizers.Adam(
#     learning_rate=learning_rate, beta_1=0.5, beta_2=0.9, epsilon=1e-08)

# inputs = tf.zeros( (2, 128, 256, 3))
# groundtruth = inputs

# 

# G = Generator(cfg)
# generator = G
# D_l = DiscriminatorLocal()
# D_g = DiscriminatorGlobal()
# loss = Loss(cfg)

# with tf.GradientTape() as g_G, tf.GradientTape() as g_D:
#     g_G.watch(groundtruth)
#     g_D.watch(groundtruth)

#     # Create all weights on the first call
#     print('===== Running all models once')
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

# # tf.Tensor([ 0.48804274 -0.9505034   0.24720697], shape=(3,), dtype=float32)
# # tf.Tensor([ 0.11747289 -0.07015078  0.36418724], shape=(3,), dtype=float32)
# # tf.Tensor([-0.04279513 -0.03300458 -0.18818763], shape=(3,), dtype=float32)
# # 336
# print(recon[0][0][0])
# print(recon[16][64][64])
# print(recon[-1][-1][-1])
# print(len(G.trainable_variables) + len(D_l.trainable_variables) + len(D_g.trainable_variables))

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))


# reader = tf.train.load_checkpoint('logs/20240108/920/checkpoint/ckpt-25')

# reader.get_tensor('step/.ATTRIBUTES/VARIABLE_VALUE')

#tf.math.top_k( list(tf.train.list_variables('./badcheckpoint/ckpt-25')), k=5, sorted=True )

for var in tf.train.list_variables('./badcheckpoint/ckpt-25'):
  print(f'{var} : {var.numpy()}')



# for var in G.trainable_variables:
#     print(var)
