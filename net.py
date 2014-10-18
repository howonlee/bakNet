import numpy as np
import numpy.random as npr

class BakNet(object):
    def __init__(self):
        self.in_layer = np.zeros(7, dtype=np.float)
        self.out_layer = np.zeros(7, dtype=np.float)
        self.h_weights = npr.random((100, 7), dtype=np.float)
        self.o_weights = npr.random((7, 100), dtype=np.float)

if __name__ == "__main__":
    bnet = BakNet()
