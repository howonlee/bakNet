import cPickle
import gzip
import numpy as np
from bak_net import BakNet

def munge_pats(dataset):
    pass

if __name__ == "__main__":
    f = gzip.open("mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load()
    #squish the training set and test set into the pattern formation
    f.close()
    print train_set
    """
    train_pats = munge_pats(train_set)
    test_pats = munge_pats(test_set)
    bnet = BakNet(784, 10000, 10, train_pats=train_pats, test_pats=test_pats)
    for i in xrange(500000):
        bnet.train()
    for i in xrange(5000):
        bnet.test()
    bnet.report()
    """
