import numpy as np
import scipy.sparse as sci_sp

class RandomNet(object):
    def __init__(self):
        self.architecture_matrix = sci_sp.dok_matrix((10000,10000))
        #create the neural architecture, it should just be kronecker graph

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
    pass
