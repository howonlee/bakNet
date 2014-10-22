import numpy as np
import numpy.random as npr
import operator
import random

def argmax(ls):
    return max(enumerate(ls), key=operator.itemgetter(1))

class BakNet(object):
    def __init__(self, n_in, n_hid, n_out, pats, delta=0.1):
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
        self.in_l = pat[0] + [1]
        self.hid_l = [0] * self.n_hid
        self.out_l = [0] * self.n_out
        self.out_teach = [0] * self.n_out
        self.out_teach[pat[1]] = 1

    def train(self):
        curr_pat = random.choice(self.pats)
        self.init_pattern(curr_pat)
        for in_idx, in_val in enumerate(self.in_l):
            #this includes the bias
            for hid_idx, _ in enumerate(self.hid_l):
                self.hid_l += self.hid_wgt[hid_idx, in_idx] * in_val
        curr_hid_idx, _ = argmax(self.hid_l)
        for hid_idx, _ in enumerate(self.hid_l):
            for out_idx, _ in enumerate(self.out_l):
                self.out_l += self.out_wgt[out_idx, hid_idx] # times 1, not hid_val
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
            #this includes the bias
            for hid_idx, _ in enumerate(self.hid_l):
                self.hid_l += self.hid_wgt[hid_idx, in_idx] * in_val
        curr_hid_idx, _ = argmax(self.hid_l)
        for hid_idx, _ in enumerate(self.hid_l):
            for out_idx, _ in enumerate(self.out_l):
                self.out_l += self.out_wgt[out_idx, hid_idx] # times 1, not hid_val
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
        self.error = float(self.correct) / float(self.total)
        print "error is: ", self.error

if __name__ == "__main__":
    #xor problem
    pats = [([0,0], 0), ([0,1],1), ([1,0],1), ([1,1],1)]
    bnet = BakNet(2, 30, 2, pats)
    for i in xrange(10000):
        bnet.train()
        #print bnet, "\n"
    for i in xrange(1000):
        bnet.test()
    bnet.report()
