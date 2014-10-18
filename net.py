import numpy as np
import numpy.random as npr

class BakNet:
    def __init__(self, n_in, n_hidden, n_out):
        self.delta = 0.01
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.h_weights = npr.random((self.n_hidden, self.n_in))
        self.o_weights = npr.random((self.n_out, self.n_hidden))

        self.h_activation = np.zeros((self.n_hidden, 1), dtype=np.float)
        self.o_activation = np.zeros((self.n_out, 1), dtype=np.float)
        self.i_output = np.zeros((self.n_in, 1), dtype=np.float)

        self.h_output = np.zeros((self.n_hidden, 1), dtype=np.float)
        self.o_output = np.zeros((self.n_out), dtype=np.float)

    def train(self, input_vals, train_vals):
        pass

    def get_output(self):
        return self.o_output

if __name__ == '__main__':
    xorSet = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xorTeach = [[0], [1], [1], [0]]

    # create network
    bak_n = BakNet(2, 2, 1)

    count = 0
    while(True):
        # choose one training sample at random
        rnd = npr.randint(0,4)
        bak_n.train(xorSet[rnd], xorTeach[rnd])

        # output for verification
        print count, xorSet[rnd], bak_n.get_output()[0],
        if bak_n.get_output()[0] > 0.80 and xorTeach[rnd][0] == 1:
            print 'GOOD',
        elif bak_n.get_output()[0] > 0.80 and xorTeach[rnd][0] == 0:
            print 'BAD',
        elif bak_n.get_output()[0] < 0.20 and xorTeach[rnd][0] == 0:
            print 'GOOD',
        elif bak_n.get_output()[0] < 0.20 and xorTeach[rnd][0] == 1:
            print 'BAD',
        elif bak_n.get_output()[0] > 0.20 and bak_n.get_output()[0] < 0.8:
            print "FUCK",
        print
        count += 1
