import numpy as np
import numpy.random as npr
import random

class BakNet(object):
    def __init__(self):
        self.delta = 0.05
        self.in_l = 0
        self.hid_l = 0
        self.out_l = 1
        self.hid_wgt = random.random() #in to hidden
        self.out_wgt = random.random() #hidden to out

    def fire(self):
        curr_input = self.in_l #random input to activate
        curr_hidden = self.hid_l #random hidden layer neuron to activate
        curr_output = self.out_l #random output layer
        if curr_output == curr_input:
            pass
        else:
            self.out_wgt -= self.delta
            self.hid_wgt -= self.delta

    def __str__(self):
        fmt_str = "delta: %s\nin layer: %s\nhidden layer: %s\nout layer: %s\nhidden weights: %s\nout weights: %s\n"
        return fmt_str % (str(self.delta), str(self.in_l), str(self.hid_l), str(self.out_l), str(self.hid_wgt), str(self.out_wgt))

if __name__ == "__main__":
    bnet = BakNet()
    print bnet
    bnet.fire()
    print bnet
