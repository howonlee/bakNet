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
        munged.append((data[i], labels[i]))
    return munged

if __name__ == "__main__":
    f = gzip.open("mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    #squish the training set and test set into the pattern formation
    f.close()
    train_pats = munge_pats(train_set)
    test_pats = munge_pats(test_set)
    bnet = BakNet(784, 10000, 10, train_pats=train_pats, test_pats=test_pats)
    num_trains = 100001
    for i in xrange(num_trains):
        bnet.train()
        if i % 1000 == 0:
            print "i: %d / %d" % (i, num_trains)
    num_tests = 5001
    for j in xrange(num_tests):
        bnet.test()
        if j % 1000 == 0:
            print "j: %d / %d" % (j, num_tests)
    bnet.report()
