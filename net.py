import numpy as np
import numpy.random as npr
import random

class BakNet(object):
    def __init__(self):
        self.delta = 0.05
        self.n_in = 2
        self.n_hid = 3
        self.n_out = 2
        self.in_l = np.zeros(self.n_in)
        self.hid_l = np.zeros(self.n_hid)
        self.out_l = np.ones(self.n_out)
        self.out_l[1] = 0
        self.hid_wgt = npr.random((self.n_hid, self.n_in)) #3 rows 2 cols
        self.out_wgt = npr.random((self.n_out, self.n_hid)) #2 rows 3 cols

    def fire(self):
        curr_input = random.randint(0,self.in_l.shape[0]-1) #current input _index_
        curr_hidden = np.argmax(self.hid_wgt[:,curr_input])
        curr_output_idx = np.argmax(self.out_wgt[:, curr_hidden])
        curr_output = self.out_l[curr_output_idx]
        if curr_output == curr_input:
            pass
        else:
            self.out_wgt[curr_output_idx, curr_hidden] -= self.delta
            self.hid_wgt[curr_hidden, curr_input] -= self.delta

    def __str__(self):
        fmt_str = "delta: %s\nin layer: %s\nhidden layer: %s\nout layer: %s\nhidden weights: %s\nout weights: %s\n"
        return fmt_str % (str(self.delta), str(self.in_l), str(self.hid_l), str(self.out_l), str(self.hid_wgt), str(self.out_wgt))

if __name__ == "__main__":
    bnet = BakNet()
    print bnet
    for i in xrange(100):
        bnet.fire()
    print bnet
