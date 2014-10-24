import numpy as np
import numpy.random as npr
import operator
import random

class BakNet(object):
    def __init__(self, n_in, n_hid, n_out, train_pats, test_pats=None, delta=0.00001):
        """
        @param n_in number of input units, not including bias
        @param n_hid number of hidden units
        This can be a tuple in a deep net, a single number for a normal ff bak net
        @param n_out number of possible output _states_
        This is important! If you have 100 possible output states, n_out will be 100!
        @param train_pats patterns to store in the net and use to train it
        @param test_pats patterns to store in the net and use to test it. if null, test_pats == train_pats. you know why this is bad, right?
        """
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
        ### this is wrong
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
        print "hidden weight: ", self.hid_wgts
        print "out weight: ", self.out_wgt
        print "hidden layer: ", self.hid_ls
        print "out layer: ", self.out_l
        print "==========================================="

    def train(self):
        curr_pat = random.choice(self.train_pats)
        self.init_pattern(curr_pat)
        #input to first hidden. this is the only one needed in shallow net
        self.hid_ls[0] = np.dot(self.hid_wgts[0], self.in_l)
        max_hid_idxs = [np.argmax(self.hid_ls[0])]
        if self.is_deep:
            for layer in xrange(1,self.n_hid_layers):
                self.hid_ls[layer] = np.dot(self.hid_wgts[layer], self.hid_ls[layer-1])
                max_hid_idxs.append(np.argmax(self.hid_ls[layer]))
        for out_idx in xrange(self.n_out):
            self.out_l[out_idx] += self.out_wgt[out_idx, max_hid_idxs[-1]]
        max_out_idx = np.argmax(self.out_l)
        if self.out_teach[max_out_idx] == 1:
            pass
        else:
            for layer_idx, hid_idx in enumerate(max_hid_idxs):
                self.hid_wgts[layer_idx][hid_idx, :] -= self.delta
            self.out_wgt[max_out_idx, max_hid_idxs[-1]] -= self.delta

    def test(self):
        curr_pat = random.choice(self.test_pats)
        self.init_pattern(curr_pat)
        self.hid_ls[0] = np.dot(self.hid_wgts[0], self.in_l)
        max_hid_idxs = [np.argmax(self.hid_ls[0])]
        if self.is_deep:
            for layer in xrange(1,self.n_hid_layers):
                self.hid_ls[layer] = np.dot(self.hid_wgts[layer], self.hid_ls[layer-1])
                max_hid_idxs.append(np.argmax(self.hid_ls[layer]))
        for out_idx in xrange(self.n_out):
            self.out_l[out_idx] += self.out_wgt[out_idx, max_hid_idxs[-1]]
        max_out_idx = np.argmax(self.out_l)
        corr_out_idx = np.argmax(self.out_teach)
        #print max_out_idx, corr_out_idx, max_hid_idxs
        if self.out_teach[max_out_idx] == 1:
            self.correct += 1
        self.total += 1

    def report(self):
        """
        Eventually we will add cross-validation and stuff like that
        """
        print "correct is: ", self.correct
        print "total is: ", self.total
        self.error = 1 - float(self.correct) / float(self.total)
        print "error is: ", self.error

if __name__ == "__main__":
    #xor problem
    pats = [(np.array([0,0]), 0), (np.array([0,1]),1), (np.array([1,0]),1), (np.array([1,1]),0)]
    bnet = BakNet(2, (10,10), 2, train_pats=pats)
    for i in xrange(50000):
        bnet.train()
    bnet.print_net()
    for i in xrange(500):
        bnet.test()
    bnet.report()
