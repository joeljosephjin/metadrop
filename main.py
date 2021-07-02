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

  # meta-training
  optim = tf.train.AdamOptimizer(tf.convert_to_tensor(args.meta_lr))

  grad_and_vars0 = [((None if grad is None else tf.clip_by_value(grad, -3.0, 3.0)), var) for grad, var in zip(net_grads[0], net_weights[0])]
  grad_and_vars1 = [((None if grad is None else tf.clip_by_value(grad, -3.0, 3.0)), var) for grad, var in zip(net_grads[1], net_weights[1])]

  # print('before: weights:', net_weights[0][0][0][0][0][0].numpy(),'grads:', net_grads[0][0][0][0][0][0].numpy())
  _ = optim.apply_gradients(grad_and_vars0)
  # print('after: weights:', net_weights[0][0][0][0][0][0].numpy(),'grads:', net_grads[0][0][0][0][0][0].numpy())
  _ = optim.apply_gradients(grad_and_vars1)
  # print('after (2): weights:', net_weights[0][0][0][0][0][0].numpy(),'grads:', net_grads[0][0][0][0][0][0].numpy())

  if i % 50 == 0: print('episode:',i*args.metabatch,'iteration:',i,'cent:',net_cent.numpy(),'acc:',net_acc_mean.numpy())
