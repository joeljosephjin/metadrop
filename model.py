from layers import *
from copy import deepcopy


class MetaDropout(tf.keras.Model):
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
    theta = {}
    for l in [1,2,3,4]:
        indim = self.input_channel if l == 1 else self.n_channel
        theta['conv%d_w'%l] = tf.get_variable('conv%d_w'%l, [3, 3, indim, self.n_channel], initializer=self.conv_init)
        theta['conv%d_b'%l] = tf.get_variable('conv%d_b'%l, [self.n_channel], initializer=self.zero_init)
    factor = 5*5 if self.dataset == 'mimgnet' else 1
    theta['dense_w'] = tf.get_variable('dense_w', [factor*self.n_channel, self.way], initializer=self.fc_init)
    theta['dense_b'] = tf.get_variable('dense_b', [self.way], initializer=self.zero_init)
    return theta

  # noise function param.
  def get_phi(self, reuse=None):
    phi = {}
    for l in [1,2,3,4]:
        indim = self.input_channel if l == 1 else self.n_channel
        phi['conv%d_w'%l] = tf.get_variable('conv%d_w'%l, [3, 3, indim, self.n_channel], initializer=self.conv_init)
        phi['conv%d_b'%l] = tf.get_variable('conb%d_b'%l, [self.n_channel], initializer=self.zero_init)
    factor = 5*5 if self.dataset == 'mimgnet' else 1
    single_w = tf.get_variable('dense_w', [factor*self.n_channel, 1], initializer=self.fc_init)
    single_b = tf.get_variable('dense_b', [1], initializer=self.zero_init)
    phi['dense_w'] = tf.tile(single_w, [1, self.way])
    phi['dense_b'] = tf.tile(single_b, [self.way])
    return phi

  # call the main network with/without perturbation
  def call(self, x, theta, phi, sample=False):
    x = tf.reshape(x, [-1, self.xdim, self.xdim, self.input_channel])

    # conventional 4-conv network --> multiplicative noise
    for l in [1,2,3,4]:
      x = conv_block(x, theta['conv%d_w'%l], theta['conv%d_b'%l], phi['conv%d_w'%l], phi['conv%d_b'%l], sample=sample, bn_scope='conv%d_bn'%l, maml=self.maml)
    # final dense layer --> additive noise
    x = dense_block(x, theta['dense_w'], theta['dense_b'], phi['dense_w'], phi['dense_b'], sample=sample, maml=self.maml)
    return x

  def metaupdate(self, theta, grads):
      theta_clone = deepcopy(theta)
      
      for key, _ in theta_clone.items():
          theta_clone[key] = theta[key]

      for key, _ in theta_clone.items():
          theta_clone[key] = theta_clone[key] - self.inner_lr * grads[key]

      return theta_clone

  def inner_function(self, data, theta, phi, losses, acc):
    xtri, ytri, xtei, ytei = data
    theta_clone = theta
    phi_clone = phi

    for i in range(self.n_steps): # 5 inner update steps
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(list(theta_clone.values()))
            inner_logits = self.call(xtri, theta_clone, phi_clone, sample=True)
            inner_loss = cross_entropy(inner_logits, ytri)

        # compute inner-gradient
        grads = inner_tape.gradient(inner_loss, list(theta_clone.values()))
        gradients = dict(zip(theta_clone.keys(), grads))
        theta_clone = self.metaupdate(theta_clone, gradients)

    logits = self.call(xtei, theta_clone, phi_clone, sample=False)
    lossi = cross_entropy(logits, ytei)
    acci = accuracy(logits, ytei)

    losses.append(lossi)
    acc.append(acci)
    return losses

  # compute the test loss over multiple tasks
  def get_loss_multiple(self, data_episode, optim):

    theta = self.theta
    phi = self.phi

    xtr, ytr, xte, yte = data_episode

    losses, acc, grads_list = [],[],[]
    # inner function
    with tf.GradientTape() as outer_tape:
        # outer_tape.watch()

        # losses = tf.map_fn()

        for xtri, ytri, xtei, ytei in zip(xtr, ytr, xte, yte):
            losses = self.inner_function(data=(xtri, ytri, xtei, ytei), theta=theta, phi=phi, losses=losses, acc=acc)

        loss_sum = sum(losses)
    # inner function

    grads = outer_tape.gradient(loss_sum, [list(theta.values()), list(phi.values())])

    grad_and_vars0 = [((None if grad is None else tf.clip_by_value(grad, -3.0, 3.0)), var) for grad, var in zip(grads[0], list(self.theta.values()))]
    grad_and_vars1 = [((None if grad is None else tf.clip_by_value(grad, -3.0, 3.0)), var) for grad, var in zip(grads[1], list(self.phi.values()))]

    _ = optim.apply_gradients(grad_and_vars0)
    _ = optim.apply_gradients(grad_and_vars1)

    return tf.reduce_mean(losses), tf.reduce_mean(acc)

