import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

log = lambda x: tf.log(x + 1e-20)
softmax = tf.nn.softmax
relu = tf.nn.relu
softplus = tf.nn.softplus
sqrt = tf.sqrt
exp = tf.exp

flatten = tf.compat.v1.layers.flatten
batchnorm = tf.keras.layers.BatchNormalization()

normal = tfp.distributions.Normal

def conv_block(x, wt, bt, wp, bp, sample=False, bn_scope='conv_bn', maml=False):
  mu = tf.nn.conv2d(x, wt, [1,1,1,1], 'SAME') + bt # NHWC
  alpha = tf.nn.conv2d(x, wp, [1,1,1,1], 'SAME') + bp # NHWC

  ones = tf.ones_like(alpha)
  mult_noise = normal(alpha, ones).sample() if sample else alpha

  if maml:
    x = mu
  else:
    x = mu * softplus(mult_noise)

  x = batchnorm(x, training=True)
  x = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'VALID')
  return x

def dense_block(x, wt, bt, wp, bp, sample=False, maml=False):
  mu = tf.matmul(flatten(x), wt) + bt
  sigma = softplus(tf.matmul(flatten(x), wp) + bp)

  if maml or not sample:
    x = mu
  else:
    x = normal(mu, sigma).sample()
  return x

def cross_entropy(logits, labels):
  return tf.compat.v1.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

def accuracy(logits, labels):
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  return tf.reduce_mean(tf.cast(correct, tf.float32))
