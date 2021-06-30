import tensorflow as tf

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
from accumulator import Accumulator
from layers import gradient_clipper

from parsers import parser


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

# for generating episode
data = Data(args)

# model object
model = MetaDropout(args)
epi = model.episodes
placeholders = [epi['xtr'], epi['ytr'], epi['xte'], epi['yte']]

# meta-training pipeline
net = model.get_loss_multiple(True)
net_cent = net['cent']
net_acc = net['acc']
net_acc_mean = tf.reduce_mean(net['acc'])
net_weights = net['weights']

# meta-training
global_step = tf.train.get_or_create_global_step()

optim = tf.train.AdamOptimizer(tf.convert_to_tensor(args.meta_lr))

if args.maml:
  var_list = [v for v in net_weights if 'phi' not in v.name]
else:
  var_list = net_weights

meta_train_op = gradient_clipper(optim, net_cent, global_step=global_step, var_list=var_list) # always returns None

config = tf.ConfigProto() # config gpu
config.gpu_options.allow_growth = True # config gpu
sess = tf.Session(config=config) # define session
sess.run(tf.global_variables_initializer()) # init variables

meta_train_logger = Accumulator('cent', 'acc') # init logger

# start training
for i in range(args.n_train_iters+1):
  episode = data.generate_episode(args, meta_training=True, n_episodes=args.metabatch)

  _, cent, acc = sess.run([meta_train_op, net_cent, net_acc_mean], feed_dict=dict(zip(placeholders, episode)))
  meta_train_logger.accum([cent, acc])

  if i % 50 == 0: meta_train_logger.print_(episode=i*args.metabatch, iteration=i)
  meta_train_logger.clear()
