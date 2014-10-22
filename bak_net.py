import numpy as np
import numpy.random as npr
import operator
import random

def argmax(ls):
    return max(enumerate(ls), key=operator.itemgetter(1))

class BakNet(object):
    def __init__(self, n_in, n_hid, n_out, pats, delta=0.001):
        """
        @param n_in number of input units, not including bias
        @param n_hid number of hidden units
        @param n_out number of possible output _states_
        @param pats patterns to store in the net
        """
        self.delta = delta
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.pats = pats
        self.in_l = None #lazily assigned
        self.hid_l = None #lazily assigned
        self.out_l = None #lazily assigned
        self.hid_wgt = npr.random((self.n_hid, self.n_in+1)) #bias
        self.out_wgt = npr.random((self.n_out, self.n_hid))
        self.correct = 0
        self.total = 0
        self.error = 0.0

    def init_pattern(self, pat):
        self.in_l = np.hstack((pat[0], np.array([1])))
        self.hid_l = np.zeros(self.n_hid)
        self.out_l = np.zeros(self.n_out)
        self.out_teach = np.zeros(self.n_out)
        self.out_teach[pat[1]] = 1

    def print_net(self):
        print "hidden weight: ", self.hid_wgt
        print "out weight: ", self.out_wgt
        print "hidden layer: ", self.hid_l
        print "out layer: ", self.out_l
        print "==========================================="

    def train(self):
        curr_pat = random.choice(self.pats)
        self.init_pattern(curr_pat)
        ###########################
        for in_idx, in_val in enumerate(self.in_l):
            for hid_idx, _ in enumerate(self.hid_l):
                self.hid_l[hid_idx] += self.hid_wgt[hid_idx, in_idx] * in_val
        curr_hid_idx, _ = argmax(self.hid_l)
        for out_idx, _ in enumerate(self.out_l):
            self.out_l[out_idx] += self.out_wgt[out_idx, curr_hid_idx] # times 1, not hid_val
        curr_out_idx, curr_out_val = argmax(self.out_l)
        if self.out_teach[curr_out_idx] == 1:
            pass
        else:
            self.hid_wgt[curr_hid_idx, :] -= self.delta
            self.out_wgt[curr_out_idx, curr_hid_idx] -= self.delta

    def test(self):
        curr_pat = random.choice(self.pats)
        self.init_pattern(curr_pat)
        for in_idx, in_val in enumerate(self.in_l):
            for hid_idx, _ in enumerate(self.hid_l):
                self.hid_l[hid_idx] += self.hid_wgt[hid_idx, in_idx] * in_val
        curr_hid_idx, _ = argmax(self.hid_l)
        for out_idx, _ in enumerate(self.out_l):
            self.out_l[out_idx] += self.out_wgt[out_idx, curr_hid_idx] # times 1, not hid_val
        curr_out_idx, curr_out_val = argmax(self.out_l)
        if self.out_teach[curr_out_idx] == 1:
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
    bnet = BakNet(2, 10, 2, pats)
    for i in xrange(50000):
        bnet.train()
    bnet.print_net()
    for i in xrange(500):
        bnet.test()
    bnet.report()
