import tensorflow as tf

import numpy as np
np.random.seed(0)

import time
import os

from model import MetaDropout
from data import Data

from parsers import parser

args = parser.parse_args()

data = Data(args)

model = MetaDropout(args)

optim = tf.keras.optimizers.Adam(tf.convert_to_tensor(args.meta_lr))

for i in range(args.n_train_iters+1):
  data_episode = data.generate_episode(args, meta_training=True, n_episodes=args.metabatch)

  loss, acc = model.get_loss_multiple(data_episode, optim)

  if i % 3 == 0: print('episode:',i*args.metabatch,'iteration:',i,'loss:',loss.numpy(),'acc:',acc.numpy())
