from layers import *
from copy import deepcopy


class MetaDropout:
  def __init__(self, args):
    self.dataset = args.dataset
    if self.dataset == 'omniglot':
      self.xdim, self.input_channel = 28, 1
      self.n_channel = 64 # channel dim of conv layers
    elif self.dataset == 'mimgnet':
      self.xdim, self.input_channel = 84, 3
      self.n_channel = 32

    self.way = args.way # num of classes per each episode
    self.n_steps = args.n_steps # num of inner gradient steps
    self.metabatch = args.metabatch # metabatch size
    self.inner_lr = args.inner_lr # inner-gradient stepsize

    # number of MC samples to evaluate the expected inner-step loss
    # over the input-dependent noise distribution
    self.n_test_mc_samp = args.n_test_mc_samp

    # whether to convert this model back to the base MAML or not
    self.maml = args.maml

    xshape = [self.metabatch, None, self.xdim*self.xdim*self.input_channel]
    yshape = [self.metabatch, None, self.way]

    # param initializers
    self.conv_init = tf.truncated_normal_initializer(stddev=0.02)
    self.fc_init = tf.random_normal_initializer(stddev=0.02)
    self.zero_init = tf.zeros_initializer()

    self.theta = self.get_theta()
    self.phi = self.get_phi()

  # main model param.
  def get_theta(self, reuse=None):
    with tf.variable_scope('theta', reuse=reuse):
      theta = {}
      for l in [1,2,3,4]:
        indim = self.input_channel if l == 1 else self.n_channel
        theta['conv%d_w'%l] = tf.get_variable('conv%d_w'%l,
            [3, 3, indim, self.n_channel], initializer=self.conv_init)
        theta['conv%d_b'%l] = tf.get_variable('conv%d_b'%l,
            [self.n_channel], initializer=self.zero_init)
      factor = 5*5 if self.dataset == 'mimgnet' else 1
      theta['dense_w'] = tf.get_variable('dense_w',
          [factor*self.n_channel, self.way], initializer=self.fc_init)
      theta['dense_b'] = tf.get_variable('dense_b',
          [self.way], initializer=self.zero_init)
      return theta

  # noise function param.
  def get_phi(self, reuse=None):
    with tf.variable_scope('phi', reuse=reuse):
      phi = {}
      for l in [1,2,3,4]:
        indim = self.input_channel if l == 1 else self.n_channel
        phi['conv%d_w'%l] = tf.get_variable('conv%d_w'%l,
            [3, 3, indim, self.n_channel], initializer=self.conv_init)
        phi['conv%d_b'%l] = tf.get_variable('conb%d_b'%l,
            [self.n_channel], initializer=self.zero_init)
      factor = 5*5 if self.dataset == 'mimgnet' else 1
      single_w = tf.get_variable('dense_w', [factor*self.n_channel, 1],
          initializer=self.fc_init)
      single_b = tf.get_variable('dense_b', [1], initializer=self.zero_init)
      phi['dense_w'] = tf.tile(single_w, [1, self.way])
      phi['dense_b'] = tf.tile(single_b, [self.way])
      return phi

  # forward the main network with/without perturbation
  def forward(self, x, theta, phi, sample=False):
    x = tf.reshape(x, [-1, self.xdim, self.xdim, self.input_channel])

    # conventional 4-conv network --> multiplicative noise
    for l in [1,2,3,4]:
      wt, bt = theta['conv%d_w'%l], theta['conv%d_b'%l]
      wp, bp = phi['conv%d_w'%l], phi['conv%d_b'%l]
      x = conv_block(x, wt, bt, wp, bp, sample=sample,
          bn_scope='conv%d_bn'%l, maml=self.maml)

    # final dense layer --> additive noise
    wt, bt = theta['dense_w'], theta['dense_b']
    wp, bp = phi['dense_w'], phi['dense_b']
    x = dense_block(x, wt, bt, wp, bp, sample=sample, maml=self.maml)
    return x

  # compute the test loss of a single task
#   def get_loss_single(self, inputs, training, reuse=None, outer_tape=None):


#     return loss, acc, grads

  # compute the test loss over multiple tasks
  def get_loss_multiple(self, training, data_episode, optim):

    xtr, ytr, xte, yte = data_episode

    losses, acc, grads_list = [],[],[]
    with tf.GradientTape() as outer_tape:
        for xtri, ytri, xtei, ytei in zip(xtr, ytr, xte, yte):
            theta = self.theta
            theta_clone = deepcopy(theta)
            phi = self.phi
            phi_clone = deepcopy(phi)

            for i in range(self.n_steps): # 5 inner update steps
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(list(theta_clone.values()))
                    inner_logits = self.forward(xtri, theta_clone, phi_clone, sample=True)
                    inner_loss = cross_entropy(inner_logits, ytri)

                # compute inner-gradient
                grads = inner_tape.gradient(inner_loss, list(theta_clone.values()))
                gradients = dict(zip(theta_clone.keys(), grads))

                theta_clone = dict(zip(theta_clone.keys(), [theta_clone[key] - self.inner_lr * gradients[key] for key in theta_clone.keys()]))

            with tf.GradientTape() as gg:
                gg.watch(theta_clone)
                gg.watch(phi_clone)
                logits = self.forward(xtei, theta_clone, phi_clone, sample=False)
                lossi = cross_entropy(logits, ytei)
            acci = accuracy(logits, ytei)

            gradsi = gg.gradient(lossi, [list(theta_clone.values()), list(phi_clone.values())])

            losses.append(lossi)
            acc.append(acci)
            grads_list.append(gradsi)

    # return the output
    theta_grads = [grads_list[i][0] for i in range(4)]
    theta_grads_sum = [None]*len(theta_grads[0])
    for i in range(len(theta_grads)):
      theta_grads_sum = [tgs+tg if tgs is not None else tg for tgs, tg in zip(theta_grads_sum, theta_grads[i])]

    phi_grads = [grads_list[i][1] for i in range(4)]
    phi_grads_sum = [None]*len(phi_grads[0])
    for i in range(len(phi_grads)):
      phi_grads_sum = [tgs+tg if tgs is not None else tg for tgs, tg in zip(phi_grads_sum, phi_grads[i])]

    grad_and_vars0 = [((None if grad is None else tf.clip_by_value(grad, -3.0, 3.0)), var) for grad, var in zip(theta_grads_sum, list(self.theta.values()))]
    grad_and_vars1 = [((None if grad is None else tf.clip_by_value(grad, -3.0, 3.0)), var) for grad, var in zip(phi_grads_sum, list(self.phi.values()))]

    _ = optim.apply_gradients(grad_and_vars0)
    _ = optim.apply_gradients(grad_and_vars1)

    return tf.reduce_mean(losses), tf.reduce_mean(acc)

