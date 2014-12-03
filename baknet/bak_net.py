import numpy as np
import numpy.random as npr
import operator
import itertools
import random
import collections
import datetime
import sys

class BakNet(object):
    def __init__(self, n_in, n_hid, n_out, train_pats, test_pats=None, delta=0.1, denoising=False):
        """
        @param n_in number of input units, not including bias
        @param n_hid number of hidden units
        This can be a tuple in a deep net, a single number for a normal ff bak net
        @param n_out number of possible output _states_
        This is important! If you have 100 possible output states, n_out will be 100!
        @param train_pats patterns to store in the net and use to train it
        @param test_pats patterns to store in the net and use to test it. if null, test_pats == train_pats. you know why this is bad, right?
        @param delta is the delta
        @param denoising whether to add noise to the data or not
        """
        self.delta = delta
        self.denoising = denoising
        self.revtabu_delta = delta / 100 #delta for reversed tabu people
        self.n_in = n_in
        self.adjusted_nin = n_in + 1 #bias
        if not isinstance(n_hid, tuple):
            n_hid = (n_hid,)
        self.n_hid = n_hid
        self.n_hid_layers = len(n_hid)
        self.is_deep = self.n_hid_layers > 1
        self.n_out = n_out
        self.train_pats = train_pats
        random.shuffle(self.train_pats)
        if test_pats == None:
            self.test_pats = train_pats
        else:
            self.test_pats = test_pats
        random.shuffle(self.test_pats)
        self.in_l = None #lazily assigned
        self.hid_ls = None #lazily assigned
        self.out_l = None #lazily assigned
        self.hid_wgts = [npr.random((self.n_hid[0], self.adjusted_nin))]
        if self.is_deep:
            for layer in xrange(self.n_hid_layers-1): #is the semantics correct here?
                self.hid_wgts.append(npr.random((self.n_hid[layer], self.n_hid[layer-1])))
        self.out_wgt = npr.random((self.n_out, self.n_hid[-1]))
        """
        Reverse Tabu
        The neurons which have given a valid value should be penalized less
        This is like a reverse version of tabu search, therefore the name
        """
        self.rev_tabu = {}
        self.train_correct = 0
        self.train_total = 0
        self.test_correct = 0
        self.test_total = 0

    def init_pattern(self, pat):
        if self.denoising:
            self.in_l = np.hstack((pat[0] + 0.02 * npr.random(pat[0].shape), np.array([1])))
        else:
            self.in_l = np.hstack((pat[0], np.array([1]))) #bias
        self.hid_ls = [np.zeros(self.n_hid[x]) for x in xrange(self.n_hid_layers)]
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
        curr_pat = self.train_pats.popleft()
        self.train_pats.append(curr_pat)
        self.init_pattern(curr_pat)
        #input to first hidden. this is the only one needed in shallow net
        self.hid_ls[0] = np.dot(self.hid_wgts[0], self.in_l)
        max_hid_idxs = [np.argmax(self.hid_ls[0])]
        min_hid_idxs = [np.argmin(self.hid_ls[0])]
        if self.is_deep:
            for layer in xrange(1, self.n_hid_layers):
                #self.hid_ls[layer] = np.dot(self.hid_wgts[layer], self.hid_ls[layer-1])
                #max_hid_idxs.append(np.argmax(self.hid_ls[layer]))
                max_hid_idxs.append(np.argmax(self.hid_wgts[layer][:, max_hid_idxs[layer-1]]))
                min_hid_idxs.append(np.argmin(self.hid_wgts[layer][:, min_hid_idxs[layer-1]]))
        max_out_idx = np.argmax(self.out_wgt[:,max_hid_idxs[-1]])
        min_out_idx = np.argmin(self.out_wgt[:,min_hid_idxs[-1]])
        #print curr_pat[1], max_out_idx
        curr_delta = self.delta / 2
        if self.out_teach[max_out_idx] == 1:
            self.rev_tabu[(max_out_idx, itertools.chain(max_hid_idxs))] = True
            self.train_correct += 1
            """
            self.hid_wgts[0][max_hid_idxs[0], :] += curr_delta
            if self.is_deep:
                for layer_idx in xrange(1, self.n_hid_layers):
                    self.hid_wgts[layer_idx][max_hid_idxs[layer_idx], max_hid_idxs[layer_idx-1]] += curr_delta
            self.out_wgt[max_out_idx, max_hid_idxs[-1]] += curr_delta
            """
        else:
            if (max_out_idx, itertools.chain(max_hid_idxs)) in self.rev_tabu:
                curr_delta = self.revtabu_delta
            self.hid_wgts[0][max_hid_idxs[0], :] -= curr_delta
            self.hid_wgts[0][min_hid_idxs[0], :] += (curr_delta * 0.8)
            if self.is_deep:
                for layer_idx in xrange(1, self.n_hid_layers):
                    self.hid_wgts[layer_idx][max_hid_idxs[layer_idx], max_hid_idxs[layer_idx-1]] -= curr_delta
                    self.hid_wgts[layer_idx][min_hid_idxs[layer_idx], min_hid_idxs[layer_idx-1]] += (curr_delta * 0.8)
            self.out_wgt[max_out_idx, max_hid_idxs[-1]] -= curr_delta
            self.out_wgt[min_out_idx, min_hid_idxs[-1]] += (curr_delta * 0.8)
        self.train_total += 1

    def test(self):
        curr_pat = self.test_pats.popleft()
        self.test_pats.append(curr_pat)
        self.init_pattern(curr_pat)
        self.hid_ls[0] = np.dot(self.hid_wgts[0], self.in_l)
        max_hid_idxs = [np.argmax(self.hid_ls[0])]
        if self.is_deep:
            for layer in xrange(1, self.n_hid_layers):
                #max_hid_idxs.append(np.argmax(self.hid_wgts[layer][:, max_hid_idxs[layer-1]]))
                self.hid_ls[layer] = np.dot(self.hid_wgts[layer], self.hid_ls[layer-1])
                max_hid_idxs.append(np.argmax(self.hid_ls[layer]))
        max_out_idx = np.argmax(self.out_wgt[:,max_hid_idxs[-1]])
        #print max_hid_idxs, curr_pat
        #print curr_pat[1], max_out_idx
        if self.out_teach[max_out_idx] == 1:
            self.rev_tabu[(max_out_idx, itertools.chain(max_hid_idxs))] = True
            self.test_correct += 1
        self.test_total += 1

    def reset_totals(self):
        self.train_correct = 0
        self.test_correct = 0
        self.train_total = 0
        self.test_total = 0

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
            error = 1 - float(self.test_correct) / float(self.test_total)
            if error < stop:
                still_training = False
            else:
                i += 1
                print "i: %d, time: %s" % (i, str(datetime.datetime.now()))
                self.report()
                self.reset_totals()

    def report(self):
        """
        Eventually we will add cross-validation and stuff like that
        """
        train_error = 1 - float(self.train_correct) / float(self.train_total)
        print train_error
        test_error = 1 - float(self.test_correct) / float(self.test_total)
        print test_error

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

def parity_problem(bits=2, report=True):
    """
    parity bit problem
    """
    bits_ls = [map(int, seq) for seq in itertools.product("01", repeat=bits)]
    pats = collections.deque(map(lambda x: (np.array(x), sum(x) % 2), bits_ls))
    #depth = tuple([10 for x in xrange(bits)])
    bnet = BakNet(bits, 1000, 2, train_pats=pats)
    #train now
    bnet.train_until()
    for i in xrange(500):
        bnet.test()
    if report:
        bnet.print_net()
        bnet.report()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "xor":
            xor_problem()
    else:
        for x in [10]:
            parity_problem(bits=x)
