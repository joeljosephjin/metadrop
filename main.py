from __future__ import print_function
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

import wandb

from parsers import parser


args = parser.parse_args()

# incorporate wandb
wandb.init(project='metadrop', entity='joeljosephjin', config=vars(args))

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

if not os.path.isdir(wandb.run.dir):
  os.makedirs(wandb.run.dir)

# for generating episode
data = Data(args)

# print(data)

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

# meta-testing pipeline
tnet = model.get_loss_multiple(False)
tnet_cent = tnet['cent']
tnet_acc = tnet['acc']
tnet_acc_mean = tf.reduce_mean(tnet['acc'])
tnet_weights = tnet['weights']

# meta-training
global_step = tf.train.get_or_create_global_step()
# print(global_step)

lr = tf.convert_to_tensor(args.meta_lr)

optim = tf.train.AdamOptimizer(lr)

if args.maml:
  var_list = [v for v in net_weights if 'phi' not in v.name]
else:
  var_list = net_weights

meta_train_op = gradient_clipper(optim, net_cent, clip=[-3., 3.],
    global_step=global_step, var_list=var_list)

# print('meta train op',meta_train_op)

saver = tf.train.Saver(tf.trainable_variables())
logfile = open(os.path.join(wandb.run.dir, 'meta_train.log'), 'w')

argdict = vars(args)
print(argdict)
for k, v in argdict.items():
    logfile.write(k + ': ' + str(v) + '\n')
logfile.write('\n')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

meta_train_logger = Accumulator('cent', 'acc')
meta_train_to_run = [meta_train_op, net_cent, net_acc_mean]
# print('sdfsd', meta_train_to_run)

meta_test_logger = Accumulator('cent', 'acc')
# print(meta_test_logger)
meta_test_to_run = [tnet_cent, tnet_acc_mean]
# print(meta_test_to_run)

start = time.time()
for i in range(args.n_train_iters+1):
  episode = data.generate_episode(args, meta_training=True,
      n_episodes=args.metabatch)

  # print('global_step', tf.train.global_step(sess, global_step))
  # itemindex = np.where(episode[0] == 1)
  # print('episode:', itemindex[0][0], itemindex[1][0], itemindex[2][0])
  fd_mtr = dict(zip(placeholders, episode))
  # print('fd_mtr:', fd_mtr)
  meta_train_logger.accum(sess.run(meta_train_to_run, feed_dict=fd_mtr))

  if i % 50 == 0:
    line = 'Iter %d start, learning rate %f' % (i, sess.run(lr))    
    print('\n' + line)
    meta_train_logger.print_(header='meta_train', episode=i*args.metabatch,
        time=time.time()-start, logfile=logfile)
      
  meta_train_logger.clear()

logfile.close()

