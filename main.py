from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import time
import os

from model import MetaDropout
from data import Data
from layers import get_train_op

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--mode', type=str, default='meta_train')
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--n_train_iters', type=int, default=60000)
parser.add_argument('--n_test_iters', type=int, default=1000)
parser.add_argument('--dataset', type=str, default='omniglot')
parser.add_argument('--way', type=int, default=20)
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--query', type=int, default=5)
parser.add_argument('--metabatch', type=int, default=16)
parser.add_argument('--meta_lr', type=float, default=1e-3)
parser.add_argument('--inner_lr', type=float, default=0.1)
parser.add_argument('--n_steps', type=int, default=5)
parser.add_argument('--n_test_mc_samp', type=int, default=1)
parser.add_argument('--maml', action='store_true', default=False)
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
net_loss = net['cent']
net_acc_mean = tf.reduce_mean(net['acc'])
net_weights = net['weights']

# meta-training
global_step = tf.train.get_or_create_global_step()
lr = tf.convert_to_tensor(args.meta_lr)
optim = tf.train.AdamOptimizer(lr)

grad_and_vars = optim.compute_gradients(net_loss, var_list=net_weights)
grad_and_vars = [((None if grad is None else tf.clip_by_value(grad, -3., 3.)), var) for grad, var in grad_and_vars]
train_operation = optim.apply_gradients(grad_and_vars, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

start = time.time()
for i in range(args.n_train_iters+1):
    episode = data.generate_episode(args, meta_training=True, n_episodes=args.metabatch)

    _, _, acc = sess.run([train_operation, net_loss, net_acc_mean], feed_dict=dict(zip(placeholders, episode)))

    if i % 50 == 0:
        print("time:", int(time.time()-start), 's, acc:', int(acc*100), '%')
