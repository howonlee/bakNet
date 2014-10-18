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
            print "train_idx: %d, train_val: %d" % (train_idx, train_val)
            for input_idx, input_val in enumerate(input_vals):
                print "input_idx: %d, input_val: %d" % (input_idx, input_val)
                self.i_output[input_idx] = input_val
                hidden_idx = np.argmax(self.h_weights[input_idx])
                print "hidden_idx: %d" % hidden_idx
                """
                print "o_output: ", self.o_output
                print "i_output: ", self.i_output
                print "h_weights: ", self.h_weights
                print "o_weights: ", self.o_weights
                """
                output_idx = np.argmax(self.o_weights)
                print "output_idx: %d" % output_idx
                self.o_output[output_idx] = self.i_output[input_idx] * self.h_weights[hidden_idx, input_idx] * self.o_weights[output_idx, hidden_idx]
                if self.o_output[output_idx] == train_val:
                    pass
                else:
                    self.o_weights[hidden_idx, output_idx] -= self.delta
                    self.h_weights[input_idx, hidden_idx] -= self.delta
                    self.o_weights[hidden_idx] += self.delta / self.n_out
                    self.h_weights[input_idx] += self.delta / self.n_hidden

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
