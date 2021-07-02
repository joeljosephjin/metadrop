import tensorflow as tf

config = tf.ConfigProto() # config gpu
config.gpu_options.allow_growth = True # config gpu

tf.enable_eager_execution(config=config)

# tf.set_random_seed(0)
# tf.random.set_random_seed(0)

import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
np.random.seed(0)

import time
import os

from model import MetaDropout
from data import Data
# from layers import gradient_clipper

from parsers import parser

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

# for generating episode
data = Data(args)

# model object
model = MetaDropout(args)

# for gradient clipping
def gradient_clipper(optim, loss, global_step=None, clip=[-3., 3.], var_list=None, net_grads=None):

  return train_op

# start training
for i in range(args.n_train_iters+1):
  data_episode = data.generate_episode(args, meta_training=True, n_episodes=args.metabatch)

  # meta-training pipeline
  net = model.get_loss_multiple(True, data_episode)
  net_cent = net['cent']
  net_acc = net['acc']
  net_acc_mean = tf.reduce_mean(net['acc'])
  net_weights = net['weights']
  net_grads = net['grads']

  # print('net gg:', net_gg)

  # meta-training
  global_step = tf.train.get_or_create_global_step()

  optim = tf.train.AdamOptimizer(tf.convert_to_tensor(args.meta_lr))

  print(len(net_grads[0]))
  print(len(net_weights[0]))

  grad_and_vars0 = [((None if grad is None else tf.clip_by_value(grad, -3.0, 3.0)), var) for grad, var in zip(net_grads[0], net_weights[0])]
  grad_and_vars1 = [((None if grad is None else tf.clip_by_value(grad, -3.0, 3.0)), var) for grad, var in zip(net_grads[1], net_weights[1])]

  _ = optim.apply_gradients(grad_and_vars0, global_step=global_step)
  _ = optim.apply_gradients(grad_and_vars1, global_step=global_step)

  if i % 50 == 0: print('episode:',i*args.metabatch,'iteration:',i,'cent:',net_cent,'acc:',net_acc_mean)
