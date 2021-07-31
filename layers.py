import tensorflow as tf
import numpy as np

# functions
log = lambda x: tf.log(x + 1e-20)
softmax = tf.nn.softmax
relu = tf.nn.relu
softplus = tf.nn.softplus
sqrt = tf.sqrt
exp = tf.exp

# layers
flatten = tf.layers.flatten
batch_norm = tf.contrib.layers.batch_norm

# blocks
def conv_block(x, wt, bt, wp, bp, sample=False, bn_scope='conv_bn', maml=False):
    mu = tf.nn.conv2d(x, wt, [1,1,1,1], 'SAME') + bt # NHWC
    alpha = tf.nn.conv2d(x, wp, [1,1,1,1], 'SAME') + bp # NHWC

    ones = tf.ones_like(alpha)
    mult_noise = tf.distributions.Normal(alpha, ones).sample() if sample else alpha
    x = mu * softplus(mult_noise)

    x = batch_norm(x, activation_fn=relu, scope=bn_scope, reuse=tf.AUTO_REUSE)
    x = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'VALID')
    return x

def dense_block(x, wt, bt, wp, bp, sample=False, maml=False):
    mu = tf.matmul(flatten(x), wt) + bt
    sigma = softplus(tf.matmul(flatten(x), wp) + bp)

    x =  tf.distributions.Normal(mu, sigma).sample()
    return x

# training modules
def cross_entropy(logits, labels):
    return tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

def accuracy(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

