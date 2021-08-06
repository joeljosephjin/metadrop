from layers import *

class MetaDropout:
    def __init__(self, args):
        self.dataset = args.dataset
        self.xdim, self.input_channel = 28, 1
        self.n_channel = 64 # channel dim of conv layers

        self.way = args.way # num of classes per each episode
        self.n_steps = args.n_steps # num of inner gradient steps
        self.metabatch = args.metabatch # metabatch size
        self.inner_lr = args.inner_lr # inner-gradient stepsize

        # whether to convert this model back to the base MAML or not
        self.maml = args.maml

        xshape = [self.metabatch, None, self.xdim*self.xdim*self.input_channel]
        yshape = [self.metabatch, None, self.way]
        # episode placeholder. 'tr': training, 'te': test
        self.episodes = {
                'xtr': tf.placeholder(tf.float32, xshape, name='xtr'),
                'ytr': tf.placeholder(tf.float32, yshape, name='ytr'),
                'xte': tf.placeholder(tf.float32, xshape, name='xte'),
                'yte': tf.placeholder(tf.float32, yshape, name='yte')}

    # main model param.
    def get_theta(self, reuse=None):
        with tf.variable_scope('theta', reuse=reuse):
            theta = {}
            for l in [1,2,3,4]:
                indim = self.input_channel if l == 1 else self.n_channel
                theta['conv%d_weight'%l] = tf.get_variable('conv%d_weight'%l, [3, 3, indim, self.n_channel])
                theta['conv%d_bias'%l] = tf.get_variable('conv%d_bias'%l, [self.n_channel])
            theta['dense_weight'] = tf.get_variable('dense_weight', [self.n_channel, self.way])
            theta['dense_bias'] = tf.get_variable('dense_bias', [self.way])
            return theta

    # noise function param.
    def get_phi(self, reuse=None):
        with tf.variable_scope('phi', reuse=reuse):
            phi = {}
            for l in [1,2,3,4]:
                indim = self.input_channel if l == 1 else self.n_channel
                phi['conv%d_weight'%l] = tf.get_variable('conv%d_weight'%l, [3, 3, indim, self.n_channel])
                phi['conv%d_bias'%l] = tf.get_variable('conb%d_bias'%l, [self.n_channel])
            single_w = tf.get_variable('dense_weight', [self.n_channel, 1])
            single_b = tf.get_variable('dense_bias', [1])
            phi['dense_weight'] = tf.tile(single_w, [1, self.way])
            phi['dense_bias'] = tf.tile(single_b, [self.way])
            return phi

    # forward the main network with/without perturbation
    def forward(self, x, theta, phi, sample=False):
        x = tf.reshape(x, [-1, self.xdim, self.xdim, self.input_channel])

        # conventional 4-conv network --> multiplicative noise
        for l in [1,2,3,4]:
            wt, bt = theta['conv%d_weight'%l], theta['conv%d_bias'%l]
            wp, bp = phi['conv%d_weight'%l], phi['conv%d_bias'%l]
            x = conv_block(x, wt, bt, wp, bp, sample=sample, bn_scope='conv%d_bn'%l, maml=self.maml)

        # final dense layer --> additive noise
        wt, bt = theta['dense_weight'], theta['dense_bias']
        wp, bp = phi['dense_weight'], phi['dense_bias']
        x = dense_block(x, wt, bt, wp, bp, sample=sample, maml=self.maml)
        return x

    # compute the test loss of a single task
    def get_loss_single(self, inputs, training, reuse=None):
        xtr, ytr, xte, yte = inputs
        theta = self.get_theta(reuse=reuse)
        phi = self.get_phi(reuse=reuse)

        # perform a few (e.g. 5) inner-gradient steps
        for i in range(self.n_steps):
            inner_loss = []

            inner_logits = self.forward(xtr, theta, phi, sample=True)
            inner_loss.append(cross_entropy(inner_logits, ytr))
            inner_loss = tf.reduce_mean(inner_loss)

            # compute inner-gradient
            grads = tf.gradients(inner_loss, list(theta.values()))
            gradients = dict(zip(theta.keys(), grads))

            # perform the current gradient step
            theta = dict(zip(theta.keys(), [theta[key] - self.inner_lr * gradients[key] for key in theta.keys()]))

        logits = self.forward(xte, theta, phi, sample=False)
        cent = cross_entropy(logits, yte)
        acc = accuracy(logits, yte)
        return cent, acc

    # compute the test loss over multiple tasks
    def get_loss_multiple(self, training):
        xtr, ytr = self.episodes['xtr'], self.episodes['ytr']
        xte, yte = self.episodes['xte'], self.episodes['yte']

        get_single = lambda inputs: self.get_loss_single(inputs, True, reuse=False)

        loss, acc = tf.map_fn(get_single, elems=(xtr, ytr, xte, yte), dtype=(tf.float32, tf.float32))
        # loss, acc = 0, 0
        # loss_, acc_ = self.get_loss_single(inputs=(xtr[0], ytr[0], xte[0], yte[0]), training=True, reuse=False)
        # loss += loss_
        # acc += acc_
        # loss_, acc_ = self.get_loss_single(inputs=(xtr[1], ytr[1], xte[1], yte[1]), training=True, reuse=True)
        # loss += loss_
        # acc += acc_
        # loss_, acc_ = self.get_loss_single(inputs=(xtr[2], ytr[2], xte[2], yte[2]), training=True, reuse=True)
        # loss += loss_
        # acc += acc_
        # loss_, acc_ = self.get_loss_single(inputs=(xtr[3], ytr[3], xte[3], yte[3]), training=True, reuse=True)
        # loss += loss_
        # acc += acc_

        # return the output
        net = {}
        net['loss'] = tf.reduce_mean(loss)
        net['acc'] = acc
        net['weights'] = tf.trainable_variables()
        return net
