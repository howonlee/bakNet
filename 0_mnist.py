import cPickle
import gzip
import numpy as np
from bak_net import BakNet
from datetime import datetime

def munge_pats(dataset):
    """
    Instead of using the mnist dataset's formatting of the numpy things,
    we munge towards python lists
    why? we are lazy
    """
    data, labels = dataset
    munged = []
    for i in xrange(data.shape[0]):
        munged.append((data[i], labels[i]))
    return munged

if __name__ == "__main__":
    """
    Simple current benchmark as of 10/22/14: 1 layer of 10000 hidden units, 10001 training spikes, 5000 tests, get ~750 right answers in about 50sec - 0.85 error, pretty dismal
    The theory is that there might be a critical state (a la critical states for ising models) for these problems. the model will stop at such a critical state and hopefully work really well
    """
    with gzip.open("mnist.pkl.gz", "rb") as f:
        train_set, valid_set, test_set = cPickle.load(f)
    train_pats = munge_pats(train_set)
    test_pats = munge_pats(test_set)
    layers = (10000,10000,10000,10000,10000)
    bnet = BakNet(784, layers, 10, train_pats=train_pats, test_pats=test_pats)
    bnet.train_until()
    num_tests = 500
    for j in xrange(num_tests):
        bnet.test()
    bnet.report()
