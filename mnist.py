import cPickle
import gzip
import numpy as np
from bak_net import BakNet

if __name__ == "__main__":
    f = gzip.open("mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load()
    #squish the training set and test set into the pattern formation
    f.close()
    bnet = BakNet(784, 10000, 10, pats)
    for i in xrange(500000):
        bnet.train()
    for i in xrange(5000):
        bnet.test()
    bnet.report()
