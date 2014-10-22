import cPickle
import gzip
import numpy as np
from bak_net import BakNet

def munge_pats(dataset):
    """
    Instead of using the mnist dataset's formatting of the numpy things,
    we munge towards python lists
    why? we are lazy
    """
    data, labels = dataset
    munged = []
    for i in xrange(data.shape[0]):
        munged.append(i,
    pass

if __name__ == "__main__":
    f = gzip.open("mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    #squish the training set and test set into the pattern formation
    f.close()
    print train_set[0][4].shape, train_set[1][4].shape
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