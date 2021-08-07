import tensorflow as tf

# config = tf.ConfigProto() # config gpu
# config.gpu_options.allow_growth = True # config gpu

# tf.enable_eager_execution(config=config)

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

# meta-training
optim = tf.optimizers.Adam(tf.convert_to_tensor(args.meta_lr))

# start training
for i in range(args.n_train_iters+1):
  data_episode = data.generate_episode(args, meta_training=True, n_episodes=args.metabatch)

  # meta-training pipeline
  loss, acc = model.get_loss_multiple(data_episode, optim)

  if i % 3 == 0: print('episode:',i*args.metabatch,'iteration:',i,'loss:',loss.numpy(),'acc:',acc.numpy())
