import numpy as np
import numpy.random as npr

class BakNet(object):
    def __init__(self):
        self.n_in = 8
        self.n_out = 8
        self.n_hidden = 100
        self.in_layer = np.zeros(n_in, dtype=np.float)
        self.out_layer = np.zeros(n_out, dtype=np.float)
        self.h_weights = npr.random((n_hidden, n_in), dtype=np.float)
        self.o_weights = npr.random((n_out, n_hidden), dtype=np.float)

if __name__ == "__main__":
    bnet = BakNet()
