import cPickle
import gzip
import numpy as np

f = gzip.open("mnist.pkl.gz", "rb")
train_set, valid_set, test_set = cPickle.load()
f.close()
