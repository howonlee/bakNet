import numpy as np
import numpy.random as npr
import scipy.sparse as sci_sp
import itertools
import random
import lib_kron

class RandomNet(object):
    def __init__(self, in_size, train_pats, test_pats=None, exponent=10):
        self.in_size = in_size
        self.train_pats = train_pats
        self.test_pats = train_pats if not test_pats else test_pats
        self.exponent = exponent
        kron_gen = lib_kron.create_generator(np.array([[0.5,0.5],[0.5,0.5]]), self.exponent)
        self.architecture_matrix = lib_kron.generate(kron_gen)
        for key in self.architecture_matrix.iterkeys():
            self.architecture_matrix[key] = npr.random()
        self.architecture_matrix.tocsc
        self.in_idxs = [x for x in xrange(self.in_size)]
        self.out_idxs = [random.randint(self.in_size+1,2**self.exponent)]
        self.nodes = npr.random(2 * 2**self.exponent)
        self.error = 0
        #there should be a convolution option and a non-convolution option

    def test(self, steps=50):
        """
        Feedforward
        """
        for x in xrange(steps):
            curr_pat = random.choice(self.test_pats)
            curr_in = curr_pat[0]
            curr_out = curr_pat[1]
            for in_idx in self.in_idxs:
                self.nodes[in_idx] = curr_in[in_idx]
            self.nodes = self.architecture_matrix.dot(self.nodes)
            print self.nodes[self.out_idxs[0]]
            self.error += self.nodes[self.out_idxs[0]] - curr_out
        #print self.error

def xor_problem():
    """
    xor problem, 10 by 10 deepish net. just as fast, as promised.
    """
    pats = [(np.array([0,0]), 0), (np.array([0,1]),1), (np.array([1,0]),1), (np.array([1,1]),0)]
    net = RandomNet(2, pats, exponent=4)
    net.test(steps=500)

if __name__ == "__main__":
    xor_problem()
