import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt

class BakNet(object):
    def __init__(self, n_in, n_hid, n_out, pats, delta=0.05):
        self.delta = delta
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.pats = pats
        self.graph = nx.Graph()
        for i in xrange(self.n_in):
            self.graph.add_node("in%d" % i, val=0)
        for i in xrange(self.n_hid):
            self.graph.add_node("hid%d" % i, val=0)
        for i in xrange(self.n_out):
            self.graph.add_node("out%d" % i, val=0)
        for i, j in itertools.product(xrange(self.n_in), xrange(self.n_hid)):
            self.graph.add_edge("in%d" % i, "hid%d" % j, weight=random.random())
        for i, j in itertools.product(xrange(self.n_hid), xrange(self.n_out)):
            self.graph.add_edge("hid%d" % i, "out%d" % j, weight=random.random())
        nx.draw(self.graph)
        plt.show()

if __name__ == "__main__":
    #xor problem
    pats = [([0,0], [0]), ([0,1],[1]), ([1,0],[1]), ([1,1],[1])]
    bnet = BakNet(5, 5, 5, pats)
    """
    print bnet
    for i in xrange(100):
        bnet.train()
        #print bnet, "\n"
    print bnet
    """
