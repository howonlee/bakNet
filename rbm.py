from __future__ import division
import numpy as np
import logging
import numpy.random as rng
import gzip
import cPickle
import sys
import datetime
import random
from scipy import optimize
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer

"""
Lief Ericsson 2011, from the py-rbm repo. Edited to be extremal optimization
"""

"""
Now, this really does have to be wrong. But how?
"""

def sigmoid(eta):
    return 1. / (1. + np.exp(-eta))

def identity(eta):
    return eta

def bernoulli(p):
    return rng.rand(*p.shape) < p

class RBM(object):
    def __init__(self, num_visible, num_hidden, scale=0.001):
        self.weights = scale * rng.randn(num_hidden, num_visible)
        self.hid_bias = scale * rng.randn(num_hidden, 1)
        self.vis_bias = scale * rng.randn(num_visible, 1)

    @property
    def num_hidden(self):
        return len(self.hid_bias)

    @property
    def num_visible(self):
        return len(self.vis_bias)

    def hidden_expectation(self, visible):
        return sigmoid(np.dot(self.weights, visible.T).T + self.hid_bias.T)

    def visible_expectation(self, hidden):
        return np.dot(hidden, self.weights) + self.vis_bias

    def iter_passes(self, visible):
        while True:
            hidden = self.hidden_expectation(visible)
            yield np.atleast_2d(visible), hidden
            visible = self.visible_expectation(bernoulli(hidden))

    def reconstruct(self, visible, passes=1):
        for i, (visible, _) in enumerate(self.iter_passes(visible)):
            if i + 1 == passes:
                return visible

class Trainer(object):
    def __init__(self, rbm):
        self.rbm = rbm
        self.grad_weights = np.zeros(rbm.weights.shape, float)
        self.grad_vis = np.zeros(rbm.vis_bias.shape, float)
        self.grad_hid = np.zeros(rbm.hid_bias.shape, float)

    def learn(self, visible):
        gradients = self.calculate_gradients(visible)
        self.punish_weight(*gradients)

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

    def punish_weight(self, weights, visible, hidden, tau=1.15):
        def update(name, g, _g):
            target = getattr(self.rbm, name)
            k = g.shape[0]
            g_len = g.shape[0]
            while k > g_len-1:
                k = int(rng.pareto(tau))
            worst = g.argsort()[-k:][::-1][-1]
            target[worst % g_len] += (rng.rand() * 0.05 - 0.025)
            if _g.shape == g.shape:
                _g[:] = np.atleast_2d(g)
            else:
                _g[:] = np.atleast_2d(g).T
        update('vis_bias', visible, self.grad_vis)
        update('hid_bias', hidden, self.grad_hid)
        update('weights', weights, self.grad_weights)

def iris_class():
    #this will require some thought
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)
    print X_train.shape
    rbm = RBM(X_train.shape[1], 2)
    rbm_trainer = Trainer(rbm)
    for row in xrange(X_train.shape[0]):
        rbm_trainer.learn(X_train[row])
    ## how to make thoughts?
    print rbm.reconstruct(X_test) - X_test

def mnist_digits():
    from scipy.io import loadmat
    data = loadmat('ex3data1.mat')
    X, y = data['X'], data['y']
    y = y.reshape(X.shape[0], )
    y = y - 1
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)
    rbm = RBM(X_train.shape[1], 20)
    rbm_trainer = Trainer(rbm)
    for row in xrange(X_train.shape[0]):
        rbm_trainer.learn(X_train[row])
    ## how to make thoughts?
    print rbm.reconstruct(X_test) - X_test

if __name__ == "__main__":
    """
    rbm = args.model and pickle.load(open(args.model, 'rb')) or Model(
        28 * 28, args.n * args.n, not args.gaussian)

    Trainer = lmj.rbm.ConvolutionalTrainer if args.conv else lmj.rbm.Trainer
    trainer = Trainer(rbm, l2=args.l2, momentum=args.momentum, target_sparsity=args.sparsity)
    """
    mnist_digits()
