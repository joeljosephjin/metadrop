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
from layers import gradient_clipper

from parsers import parser

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

# for generating episode
data = Data(args)

# model object
model = MetaDropout(args)
# epi = model.episodes
# data_placeholders = [epi['xtr'], epi['ytr'], epi['xte'], epi['yte']]



# sess = tf.Session(config=config) # define session
# sess.run(tf.global_variables_initializer()) # init variables

# start training
for i in range(args.n_train_iters+1):
  data_episode = data.generate_episode(args, meta_training=True, n_episodes=args.metabatch)

  # meta-training pipeline
  net = model.get_loss_multiple(True, data_episode)
  net_cent = net['cent']
  net_acc = net['acc']
  net_acc_mean = tf.reduce_mean(net['acc'])
  net_weights = net['weights']
  net_gg = net['grad_tapes']

  print('net gg:', net_gg)

  # meta-training
  global_step = tf.train.get_or_create_global_step()

  optim = tf.train.AdamOptimizer(tf.convert_to_tensor(args.meta_lr))

  if args.maml:
    var_list = [v for v in net_weights if 'phi' not in v.name]
  else:
    var_list = net_weights

  meta_train_operation = gradient_clipper(optim, net_cent, global_step=global_step, var_list=var_list)

  # _, cent, acc = sess.run([meta_train_operation, net_cent, net_acc_mean], feed_dict=dict(zip(data_placeholders, data_episode)))

  if i % 50 == 0: print('episode:',i*args.metabatch,'iteration:',i,'cent:',cent,'acc:',acc)
