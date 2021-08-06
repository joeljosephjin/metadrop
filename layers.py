import tensorflow as tf

# tf.set_random_seed(0)
# tf.random.set_random_seed(0)

import numpy as np
# np.random.seed(0)

# functions
log = lambda x: tf.log(x + 1e-20)
softmax = tf.nn.softmax
relu = tf.nn.relu
softplus = tf.nn.softplus
sqrt = tf.sqrt
exp = tf.exp

# layers
# flatten = tf.layers.flatten
flatten = tf.compat.v1.layers.flatten

# distribution
normal = tf.distributions.Normal

# blocks
def conv_block(x, wt, bt, wp, bp, sample=False, bn_scope='conv_bn', maml=False):
  mu = tf.nn.conv2d(x, wt, [1,1,1,1], 'SAME') + bt # NHWC
  alpha = tf.nn.conv2d(x, wp, [1,1,1,1], 'SAME') + bp # NHWC

  ones = tf.ones_like(alpha)
  mult_noise = normal(alpha, ones).sample() if sample else alpha

  if maml:
    x = mu
  else:
    x = mu * softplus(mult_noise)

#   print('1:', x[0][0][0][0].numpy())
  x = tf.contrib.layers.batch_norm(x)
#   print('2:', x[0][0][0][0].numpy())
#   x = tf.compat.v1.layers.batch_normalization(x, center=True, scale=True)
#   print('3:', x[0][0][0][0].numpy())
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

# training modules
def cross_entropy(logits, labels):
  return tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

def accuracy(logits, labels):
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  return tf.reduce_mean(tf.cast(correct, tf.float32))
