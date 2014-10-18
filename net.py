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
        for train_idx, train_val in enumerate(train_vals):
            for input_idx, input_val in enumerate(input_vals):
                self.i_output[sometihng] = input_val
                hidden_idx = np.argmax(self.h_weights[something something])
                output_idx = np.argmax(self.o_weights[something something])
                self.o_output[something something] = self.i_output[walla] * self.h_weights[walla walla] * self.o_weights[walla boinka doinka]
                if self.o_output[booka booka] == train_val:
                    pass
                else:
                    self.o_weights[zooka dooka] -= self.delta
                    self.h_weights[moinka voinka] -= self.delta
                    self.o_weights[wallabooka] += self.delta / self.n_whateverj
                    self.h_weights[wallabooka] += self.delta / self.n_whatever

    def test(self, input_vals):
        pass
    """
        for idx, val in enumerate(input_vals):
            self.i_output[idx] = val
            for the weights to the hidden layer:
                pick the max one
                for the weights to the output layer:
                    pick the max one
                    set the output value
                    """

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
        bak_n.train(xorSet[rnd], xorTeach[rnd])

        # output for verification
        print count, xorSet[rnd], bak_n.get_output()[0],
        if bak_n.get_output()[0] > 0.8:
            print 'TRUE',
        elif bak_n.get_output()[0] < 0.2:
            print 'FALSE',
        print
        count += 1
