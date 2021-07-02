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
  # print(net_gg)
  print(var_list,'here comes the loss' ,loss)
  grads = []
  for ggi in net_gg:
    grads.append(ggi.gradient(loss, var_list))

  # grad_and_vars = optim.compute_gradients(loss, var_list=var_list)
  print(grads)
  grad_and_vars = []
  if clip is not None:
      grad_and_vars = [((None if grad is None else tf.clip_by_value(grad, clip[0], clip[1])), var) for grad, var in zip(grads, var_list)]
  print(grad_and_vars)
  train_op = optim.apply_gradients(grad_and_vars, global_step=global_step)
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

  if args.maml:
    var_list = [v for v in net_weights if 'phi' not in v.name]
  else:
    var_list = net_weights

  # print('var list here', var_list)

  meta_train_operation = gradient_clipper(optim, net_cent, global_step=global_step, var_list=var_list, net_grads=net_grads)

  # _, cent, acc = sess.run([meta_train_operation, net_cent, net_acc_mean], feed_dict=dict(zip(data_placeholders, data_episode)))

  if i % 50 == 0: print('episode:',i*args.metabatch,'iteration:',i,'cent:',cent,'acc:',acc)
