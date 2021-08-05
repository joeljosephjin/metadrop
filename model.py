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
                theta['conv%d_w'%l] = tf.get_variable('conv%d_w'%l, [3, 3, indim, self.n_channel])
                theta['conv%d_b'%l] = tf.get_variable('conv%d_b'%l, [self.n_channel])
            theta['dense_w'] = tf.get_variable('dense_w', [self.n_channel, self.way])
            theta['dense_b'] = tf.get_variable('dense_b', [self.way])
            return theta

    # noise function param.
    def get_phi(self, reuse=None):
        with tf.variable_scope('phi', reuse=reuse):
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

    # forward the main network with/without perturbation
    def forward(self, x, theta, phi, sample=False):
        x = tf.reshape(x, [-1, self.xdim, self.xdim, self.input_channel])

        # conventional 4-conv network --> multiplicative noise
        for l in [1,2,3,4]:
            wt, bt = theta['conv%d_w'%l], theta['conv%d_b'%l]
            wp, bp = phi['conv%d_w'%l], phi['conv%d_b'%l]
            x = conv_block(x, wt, bt, wp, bp, sample=sample, bn_scope='conv%d_bn'%l, maml=self.maml)

        # final dense layer --> additive noise
        wt, bt = theta['dense_w'], theta['dense_b']
        wp, bp = phi['dense_w'], phi['dense_b']
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

        get_single_train = lambda inputs: self.get_loss_single(inputs, True, reuse=False)
        get_single_test = lambda inputs: self.get_loss_single(inputs, False, reuse=True)
        get_single = get_single_train if training else get_single_test

        cent, acc = tf.map_fn(get_single, elems=(xtr, ytr, xte, yte), dtype=(tf.float32, tf.float32), parallel_iterations=self.metabatch)

        # return the output
        net = {}
        net['cent'] = tf.reduce_mean(cent)
        net['acc'] = acc
        net['weights'] = tf.trainable_variables()
        return net
