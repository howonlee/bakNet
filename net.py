import numpy as np
import numpy.random as npr
import random

class BakNet(object):
    def __init__(self, n_in, n_hid, n_out, pats, delta=0.05):
        """
        Pattern formats:
        list of 2-tuples, each tuple has input and output pattern
        """
        self.delta = delta
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.pats = pats
        self.hid_l = np.zeros(self.n_hid)
        self.in_l = None #lazily assigned
        self.out_l = None #lazily assigned
        self.hid_wgt = npr.random((self.n_hid, self.n_in))
        self.out_wgt = npr.random((self.n_out, self.n_hid))

    def train(self):
        curr_pat = random.choice(self.pats)
        self.in_l = curr_pat[0]
        self.out_l = curr_pat[1]
        curr_input = random.randint(0,len(self.in_l)-1) #current input _index_
        curr_hidden = np.argmax(self.hid_wgt[:,curr_input])
        curr_output_idx = np.argmax(self.out_wgt[:, curr_hidden])
        curr_output = self.out_l[curr_output_idx]
        if curr_output == curr_input:
            pass
        else:
            self.out_wgt[curr_output_idx, curr_hidden] -= self.delta
            self.hid_wgt[curr_hidden, curr_input] -= self.delta

    def test(self):

    def __str__(self):
        fmt_str = "delta: %s\nin layer: %s\nhidden layer: %s\nout layer: %s\nhidden weights: %s\nout weights: %s\n"
        return fmt_str % (str(self.delta), str(self.in_l), str(self.hid_l), str(self.out_l), str(self.hid_wgt), str(self.out_wgt))

if __name__ == "__main__":
    pats = [([0,0], [0]), ([0,1],[1]), ([1,0],[1]), ([1,1],[1])]
    bnet = BakNet(2, 5, 1, pats)
    print bnet
    for i in xrange(100):
        bnet.fire()
    print bnet
