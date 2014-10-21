import numpy as np
import numpy.random as npr

class BakNet(object):
    def __init__(self, n_in, n_hid, n_out, pats):
        """
        @param n_in number of input units, not including bias
        @param n_hid number of hidden units
        @param n_out number of possible output _states_
        @param pats patterns to store in the net
        """
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
        self.in_l = pat[0]
        self.in_l.append(1) #bias
        self.hid_l = [0] * self.n_hid
        self.out_l = [0] * self.n_out
        self.out_l[pat[1]] = 1

    def train(self):
        curr_pat = random.choice(self.pats)
        self.init_pattern(curr_pat)
        for each thing in input:
            add up the hidden layer thing
        choose the proper output unit for hidden layer
        if it is right:
            pass
        else:
            do correction

    def test(self):
        curr_pat = random.choice(self.pats)
        self.init_pattern(curr_pat)
        for each thing in input:
            add up the hidden layer thing
        choose the proper output unit for hidden layer
        if right:
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
    bnet = BakNet(2, 3, 2, pats)
    for i in xrange(100):
        bnet.train()
        #print bnet, "\n"
    for i in xrange(100):
        bnet.test()
    bnet.report()
