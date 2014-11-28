import numpy as np
import logging
import numpy.random as rng

"""
Lief Ericsson 2011, from the py-rbm repo. Edited to be extremal optimization
"""

def sigmoid(eta):
    return 1. / (1. + np.exp(-eta))

def identity(eta):
    return eta

def bernoulli(p):
    return rng.rand(*p.shape) < p

class RBM(object):
    '''
    In this way, each
    of the visible units is independent from the other visible units when
    conditioned on the state of the hidden layer, and each of the hidden units
    is independent of the others when conditioned on the state of the visible
    layer. This conditional independence makes inference tractable for the units
    in a single RBM.

    To "encode" a signal by determining the state of the hidden units given some
    visible data ("signal"),

    1. the signal is presented to the visible units, and
    2. the states of the hidden units are sampled from the conditional
       distribution given the visible data.

    To "decode" an encoding in the hidden units,

    3. the states of the visible units are sampled from the conditional
       distribution given the states of the hidden units.

    Once a signal has been encoded and then decoded,

    4. the sampled visible units can be compared directly with the original
       visible data.

    Training takes place by presenting a number of data points to the network,
    encoding the data, reconstructing it from the hidden states, and encoding
    the reconstruction in the hidden units again. Then, using contrastive
    divergence (Hinton 2002; Hinton & Salakhutdinov 2006), the gradient is
    approximated using the correlations between visible and hidden units in the
    first encoding and the same correlations in the second encoding.
    '''

    def __init__(self, num_visible, num_hidden, binary=True, scale=0.001):
        '''Initialize a restricted boltzmann machine.

        Parameters
        ----------
        num_visible : int
            The number of visible units.

        num_hidden : int
            The number of hidden units.

        binary : bool
            True if the visible units are binary, False if the visible units are
            normally distributed.

        scale : float
            Sample initial weights from N(0, scale).
        '''
        self.weights = scale * rng.randn(num_hidden, num_visible)
        self.hid_bias = scale * rng.randn(num_hidden, 1)
        self.vis_bias = scale * rng.randn(num_visible, 1)
        self._visible = binary and sigmoid or identity

    @property
    def num_hidden(self):
        return len(self.hid_bias)

    @property
    def num_visible(self):
        return len(self.vis_bias)

    def hidden_expectation(self, visible, bias=0.):
        '''Given visible data, return the expected hidden unit values.'''
        return sigmoid(np.dot(self.weights, visible.T).T + self.hid_bias + bias)

    def visible_expectation(self, hidden, bias=0.):
        '''Given hidden states, return the expected visible unit values.'''
        return self._visible(np.dot(hidden, self.weights) + self.vis_bias + bias)

    def iter_passes(self, visible):
        '''Repeatedly pass the given visible layer up and then back down.
        Returns
        -------
        Generates a sequence of (visible, hidden) states. The first pair will be
        the (original visible, resulting hidden) states, followed by pairs
        containing the values from (visible down-pass, hidden up-pass).
        '''
        while True:
            hidden = self.hidden_expectation(visible)
            yield visible, hidden
            visible = self.visible_expectation(bernoulli(hidden))

    def reconstruct(self, visible, passes=1):
        '''Reconstruct a given visible layer through the hidden layer.

        Parameters
        ----------
        visible : ndarray
            The initial state of the visible layer.

        passes : int
            The number of up- and down-passes.

        Returns
        -------
        An array containing the reconstructed visible layer after the specified
        number of up- and down- passes.
        '''
        for i, (visible, _) in enumerate(self.iter_passes(visible)):
            if i + 1 == passes:
                return visible


class Trainer(object):
    def __init__(self, rbm):
        self.rbm = rbm
        self.grad_weights = np.zeros(rbm.weights.shape, float)
        self.grad_vis = np.zeros(rbm.vis_bias.shape, float)
        self.grad_hid = np.zeros(rbm.hid_bias.shape, float)

    def learn(self, visible, learning_rate=0.2):
        gradients = self.calculate_gradients(visible)
        self.apply_gradients(*gradients, learning_rate=learning_rate)

    def calculate_gradients(self, visible_batch):
        '''Calculate gradients for a batch of visible data.

        Returns a 3-tuple of gradients: weights, visible bias, hidden bias.

        visible_batch: A (batch size, visible units) array of visible data. Each
          row represents one visible data sample.
        '''
        passes = self.rbm.iter_passes(visible_batch)
        v0, h0 = passes.next()
        v1, h1 = passes.next()

        gw = (np.dot(h0.T, v0) - np.dot(h1.T, v1)) / len(visible_batch)
        gv = (v0 - v1).mean(axis=0)
        gh = (h0 - h1).mean(axis=0)
        logging.debug('displacement: %.3g, hidden std: %.3g',
                      np.linalg.norm(gv), h0.std(axis=1).mean())

        return gw, gv, gh

    def punish_weight(self, weights, visible, hidden):
        pass

    def apply_gradients(self, weights, visible, hidden, learning_rate=0.2):
        '''
        '''
        def update(name, g, _g, l2=0):
            target = getattr(self.rbm, name)
            g *= 1 - self.momentum
            g += self.momentum * (g - l2 * target)
            target += learning_rate * g
            _g[:] = g

        update('vis_bias', visible, self.grad_vis)
        update('hid_bias', hidden, self.grad_hid)
        update('weights', weights, self.grad_weights, self.l2)

