import numpy as np
import numpy.random as npr
import operator
import itertools
import random
import datetime
import sys

class BakNet(object):
    def __init__(self, n_in, n_hid, n_out, train_pats, test_pats=None, delta=None):
        """
        @param delta a _function_ that acts as delta
        @param n_in number of input units, not including bias
        @param n_hid number of hidden units
        This can be a tuple in a deep net, a single number for a normal ff bak net
        @param n_out number of possible output _states_
        This is important! If you have 100 possible output states, n_out will be 100!
        @param train_pats patterns to store in the net and use to train it
        @param test_pats patterns to store in the net and use to test it. if null, test_pats == train_pats. you know why this is bad, right?
        """
        if delta is None:
            self.delta = lambda: 0.0001
            #some sort of annealing procedure
        else:
            self.delta = delta
        self.n_in = n_in
        self.adjusted_nin = n_in + 1 #bias
        if isinstance(n_hid, tuple):
            self.n_hid = n_hid
            self.is_deep = True
            self.n_hid_layers = len(n_hid)
        else:
            self.n_hid = [n_hid]
            self.is_deep = False #then only one layer
            self.n_hid_layers = 1
        self.n_out = n_out
        self.train_pats = train_pats
        if test_pats == None:
            self.test_pats = train_pats
        else:
            self.test_pats = test_pats
        self.in_l = None #lazily assigned
        self.hid_ls = None #lazily assigned
        self.out_l = None #lazily assigned
        self.hid_wgts = [npr.random((self.n_hid[0], self.adjusted_nin))]
        if self.is_deep:
            for layer in xrange(self.n_hid_layers-1):
                self.hid_wgts.append(npr.random((self.n_hid[layer], self.n_hid[layer-1])))
        self.out_wgt = npr.random((self.n_out, self.n_hid[-1]))
        self.correct = 0
        self.total = 0
        self.error = 0.0

    def init_pattern(self, pat):
        self.in_l = np.hstack((pat[0], np.array([1]))) #bias
        self.hid_ls = []
        for x in xrange(self.n_hid_layers):
            self.hid_ls.append(np.zeros(self.n_hid[x]))
        self.out_l = np.zeros(self.n_out)
        self.out_teach = np.zeros(self.n_out)
        self.out_teach[pat[1]] = 1

    def print_net(self):
        """
        Prints network
        """
        if self.n_hid < 50:
            print "hidden weight: ", self.hid_wgts
            print "out weight: ", self.out_wgt
            print "hidden layer: ", self.hid_ls
        if self.n_out < 50:
            print "out layer: ", self.out_l
        if self.n_hid >= 50 and self.n_out >= 50:
            print "net is too big to print"
        print "==========================================="

    def train(self):
        curr_pat = random.choice(self.train_pats)
        self.init_pattern(curr_pat)
        #input to first hidden. this is the only one needed in shallow net
        self.hid_ls[0] = np.dot(self.hid_wgts[0], self.in_l)
        max_hid_idxs = [np.argmax(self.hid_ls[0])]
        if self.is_deep:
            for layer in xrange(1,self.n_hid_layers):
                max_hid_idxs.append(np.argmax(self.hid_wgts[layer][:,max_hid_idxs[layer-1]]))
        for out_idx in xrange(self.n_out):
            self.out_l[out_idx] += self.out_wgt[out_idx, max_hid_idxs[-1]]
        max_out_idx = np.argmax(self.out_l)
        if self.out_teach[max_out_idx] == 1:
            pass
        else:
            curr_delta = self.delta()
            self.hid_wgts[0][max_hid_idxs[0], :] -= curr_delta
            if self.is_deep:
                for layer_idx in xrange(1, self.n_hid_layers):
                    self.hid_wgts[layer_idx][max_hid_idxs[layer_idx], max_hid_idxs[layer_idx-1]] -= curr_delta
            self.out_wgt[max_out_idx, max_hid_idxs[-1]] -= curr_delta

    def test(self):
        curr_pat = random.choice(self.test_pats)
        self.init_pattern(curr_pat)
        self.hid_ls[0] = np.dot(self.hid_wgts[0], self.in_l)
        max_hid_idxs = [np.argmax(self.hid_ls[0])]
        if self.is_deep:
            for layer in xrange(1,self.n_hid_layers):
                max_hid_idxs.append(np.argmax(self.hid_wgts[layer][:,max_hid_idxs[layer-1]]))
        for out_idx in xrange(self.n_out):
            self.out_l[out_idx] += self.out_wgt[out_idx, max_hid_idxs[-1]]
        max_out_idx = np.argmax(self.out_l)
        corr_out_idx = np.argmax(self.out_teach)
        #print max_out_idx, corr_out_idx, max_hid_idxs
        if self.out_teach[max_out_idx] == 1:
            self.correct += 1
        self.total += 1

    def train_until(self, train_steps=10000, test_steps=500, stop=0.02):
        """
        Trains until correct on the test set for 100 iters
        """
        still_training = True
        i = 0
        while still_training:
            for st in xrange(train_steps):
               self.train()
            for st in xrange(test_steps):
               self.test()
            self.error = 1 - float(self.correct) / float(self.total)
            if self.error < stop:
                still_training = False
                self.correct = 0
                self.total = 0
            else:
                i += 1
                print "i: %d, time: %s" % (i, str(datetime.datetime.now()))
                self.report()
                self.correct = 0
                self.total = 0

    def report(self):
        """
        Eventually we will add cross-validation and stuff like that
        """
        print "correct is: ", self.correct
        print "total is: ", self.total
        self.error = 1 - float(self.correct) / float(self.total)
        print "error is: ", self.error

def xor_problem():
    """
    xor problem, 10 by 10 deepish net. just as fast, as promised.
    """
    pats = [(np.array([0,0]), 0), (np.array([0,1]),1), (np.array([1,0]),1), (np.array([1,1]),0)]
    bnet = BakNet(2, (10,10), 2, train_pats=pats)
    bnet.train_until(train_steps=1000)
    bnet.print_net()
    for i in xrange(500):
        bnet.test()
    bnet.report()

def parity_problem(bits=2):
    """
    parity bit problem
    """
    bits_ls = [map(int, seq) for seq in itertools.product("01", repeat=bits)]
    pats = map(lambda x: (np.array(x), sum(x) % 2), bits_ls)
    bnet = BakNet(bits, (3000,3000), 2, train_pats=pats)
    bnet.train_until()
    bnet.print_net()
    for i in xrange(500):
        bnet.test()
    bnet.report()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "xor":
            xor_problem()
    else:
        for x in xrange(2,15):
            print "WE PARITYING OVER HERE ALL THE TIME"
            print "WITH BITS = ", x
            print "================="
            print "================="
            print "================="
            parity_problem(bits=x)
