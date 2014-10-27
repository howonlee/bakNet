import numpy as np
import numpy.random as npr
import scipy.sparse as sci_sp
import itertools
import random
import lib_kron

class RandomNet(object):
    def __init__(self, train_pats, test_pats=None, exponent=10):
        self.train_pats = train_pats
        self.test_pats = train_pats if not test_pats else test_pats
        self.exponent = exponent
        kron_gen = lib_kron.create_generator(np.array([[0.5,0.5],[0.5,0.5]]), self.exponent)
        self.architecture_matrix = lib_kron.generate(kron_gen)
        for key in self.architecture_matrix.iterkeys():
            self.architecture_matrix[key] = npr.random()
        self.in_idxs = [x for x in xrange(784)]
        self.out_idxs = [random.randint(785,2**self.exponent)]
        self.nodes = npr.random(2**self.exponent)
        print self.nodes


        #pick a random set of 784 nodes with indegree 0, outdegree 1 to be in nodes
        #pick a random set of 1 node with indegree the most, outdegree 0 to be out node
        #if you fail, generate again

        #there should be a convolution option and a non-convolution option
        #weights are random
        #there should be no training

    def test(self):
        #feed forward, mon ami
        pass

if __name__ == "__main__":
    net = RandomNet()
    pass
