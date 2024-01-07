#!/usr/bin/env python
import tensorflow as tf
import tensorflow.compat.v1 as v1

import numpy as np
import tf_slim as slim
import sys

from model.generator import Generator
from model.discriminator import DiscriminatorLocal, DiscriminatorGlobal


from contextlib import contextmanager


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

tf.compat.v1.random.set_random_seed(1)
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(1)

# random_tool = v1.keras.utils.DeterministicRandomTestTool()
# with random_tool.scope():

inputs = tf.ones( (32, 128, 128, 3))

cfg = type('cfg', (), {'weight_decay': 0.00002, 'batch_size_per_gpu': 32})()

G = Generator(cfg)
D_l = DiscriminatorLocal()
D_g = DiscriminatorGlobal()

# Create all weights on the first call
print('===== Running all models once')
G(inputs)
# D_l(inputs)
# D_g(inputs)

# Verify that no new weights are created in followup calls
with assert_no_variable_creations():
  print('===== Running all models again, looking for new vars')
  G(inputs)
  # D_l(inputs)
  # D_g(inputs)
  print('===== Done')
with catch_and_raise_created_variables():
  print('Running all models one last time')
  G(inputs)
  # D_l(inputs)
  # D_g(inputs)
  print('===== Done')

print(len(G.trainable_variables) + len(D_l.trainable_variables) + len(D_g.trainable_variables))

_, recon = G(inputs)

# tf.Tensor([ 0.48804274 -0.9505034   0.24720697], shape=(3,), dtype=float32)
# tf.Tensor([ 0.11747289 -0.07015078  0.36418724], shape=(3,), dtype=float32)
# tf.Tensor([-0.04279513 -0.03300458 -0.18818763], shape=(3,), dtype=float32)
print(recon[0][0][0])
print(recon[16][64][64])
print(recon[-1][-1][-1])
