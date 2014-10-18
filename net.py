import numpy as np
import numpy.random as npr

class BakNet:
    def __init__(self, n_in, n_hidden, n_out):
        self.pi = 0.1
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.h_weights = np.zeros((self.n_hidden, self.n_in)) + 0.8
        self.o_weights = np.zeros((self.n_out, self.n_hidden)) + 0.8

        self.h_activation = np.zeros((self.n_hidden, 1), dtype=np.float)
        self.o_activation = np.zeros((self.n_out, 1), dtype=np.float)

        self.i_output = np.zeros((self.n_in, 1), dtype=np.float)
        self.h_output = np.zeros((self.n_hidden, 1), dtype=np.float)
        self.o_output = np.zeros((self.n_out), dtype=np.float)

    def forward(self, input):
        pass

    def backwards(self, teach):
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
        rnd = npr.randint(0,3)
         
        # forward and backward pass
        bak_n.forward(xorSet[rnd])
        bak_n.backward(xorTeach[rnd])
         
        # output for verification
        print count, xorSet[rnd], bak_n.getOutput()[0],
        if bak_n.getOutput()[0] > 0.8:
            print 'TRUE',
        elif bak_n.getOutput()[0] < 0.2:
            print 'FALSE',
        print    
        count += 1
