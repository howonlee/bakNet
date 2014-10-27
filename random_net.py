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
        self.tst_correct = 0
        self.tst_total = 0
        #there should be a convolution option and a non-convolution option

    def test(self):
        """
        Feedforward
        """
        pass
        self.report()

def xor_problem():
    """
    xor problem, 10 by 10 deepish net. just as fast, as promised.
    """
    pats = [(np.array([0,0]), 0), (np.array([0,1]),1), (np.array([1,0]),1), (np.array([1,1]),0)]
    net = RandomNet(train_pats=pats)
    net.test(steps=500)

if __name__ == "__main__":
    net = RandomNet()
    pass
