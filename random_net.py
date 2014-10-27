import numpy as np
import scipy.sparse as sci_sp
import lib_kron

class RandomNet(object):
    def __init__(self):
        kron_gen = lib_kron.create_generator(np.array([[0.5,0.5],[0.5,0.5]]), 6)
        self.architecture_matrix = lib_kron.generate(kron_gen)
        print self.architecture_matrix

        #pick a random set of 784 nodes with indegree 0, outdegree 1 to be in nodes
        #pick a random set of 1 node with indegree the most, outdegree 0 to be out node
        #if you fail, generate again

        #there should be a convolution option and a non-convolution option
        #weights are random
        #there should be no training
        pass

    def test(self):
        #feed forward, mon ami
        pass

if __name__ == "__main__":
    net = RandomNet()
    pass
