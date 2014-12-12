import numpy as np
import gzip
import random
import cPickle
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
"""
echen, on github
"""

class RBM:
    def __init__(self, num_visible, num_hidden, is_eo=False, learning_rate = 0.06):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)
        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)
        self.is_eo = is_eo
        self.errors = []
        if self.is_eo:
            self.best_weights = self.weights.copy()
            self.best_error = float("inf")

    def save(path):
        pass

    def load(path):
        pass

    def train(self, data, max_epochs = 1000, tau=1.0):
        """
        Train the machine.

        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the states of visible units.
        """
        num_examples = data.shape[0]
        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis = 1)

        for epoch in range(max_epochs):
            # Clamp to the data and sample from the hidden units.
            # (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
            # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
            # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:,0] = 1 # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states
            # themselves.
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Update weights.
            if self.is_eo:
                ############################################
                energies = (neg_associations - pos_associations)
                ### the contribution of each weight to the total
                #energies = (data - neg_visible_probs).sum(axis=0)
                #print "weight shape : ", self.weights.shape
                #print "energies shape: ", energies.shape

                #k = 0
                k = energies.shape[0] + 1
                while k > energies.shape[0]:
                    k = int(np.random.pareto(tau))

                #print "k: ", k
                #print "argsorted: ", energies.ravel().argsort()
                #print "k member: ", energies.ravel().argsort()[-(k+1)]
                #print "argsorted len: ", len(energies.ravel().argsort())
                worst = energies.ravel().argsort()[-(k+1)]
                #print worst
                rand_idx = random.randrange(0, energies.shape[0])
                #print "energy: ", energies.flat[worst]
                self.weights.flat[worst] -= 1
                #self.weights.flat[worst] -= np.random.rand()
                error = np.sum((data - neg_visible_probs) ** 2)
                #self.is_eo = False
                if error < self.best_error:
                    self.best_error = error
                    self.best_weights = self.weights.copy()
                self.errors.append(error)
            else:
                self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)
                error = np.sum((data - neg_visible_probs) ** 2)
                self.errors.append(error)
            print("Epoch %s: error is %s" % (epoch, error))

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible units.

        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the hidden units (plus a bias unit)
        # sampled from a training example.
        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis = 1)

        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(data, self.weights)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._logistic(hidden_activations)
        # Turn the hidden units on with their specified probabilities.

        ## turn this one on again when done - howon

        hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        # Always fix the bias unit to 1.
        # hidden_states[:,0] = 1

        # Ignore the bias units.
        hidden_states = hidden_states[:,1:]
        return hidden_states

    def run_hidden(self, data):
        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the visible units (plus a bias unit)
        # sampled from a training example.
        visible_states = np.ones((num_examples, self.num_visible + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis = 1)

        # Calculate the activations of the visible units.
        visible_activations = np.dot(data, self.weights.T)
        # Calculate the probabilities of turning the visible units on.
        visible_probs = self._logistic(visible_activations)
        # Turn the visible units on with their specified probabilities.
        visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
        # Ignore the bias units.
        visible_states = visible_states[:,1:]
        return visible_states

    def daydream(self, num_samples):
        """
        samples: A matrix, where each row is a sample of the visible units produced while the network was
        daydreaming.
        """
        samples = np.ones((num_samples, self.num_visible + 1))
        # Take the first sample from a uniform distribution.
        samples[0,1:] = np.random.rand(self.num_visible)

        for i in range(1, num_samples):
            visible = samples[i-1,:]
            hidden_activations = np.dot(visible, self.weights)
            hidden_probs = self._logistic(hidden_activations)
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            hidden_states[0] = 1

            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            #visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i,:] = visible_probs

        return samples[:,1:]

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

if __name__ == '__main__':
    digits = datasets.load_digits()
    X = np.asarray(digits.data, "float32")
    X, Y = nudge_dataset(X, digits.target)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001) #scale
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print X_train.shape
    r = RBM(num_visible = 64, num_hidden = 100, is_eo=True)
    r.train(X_train, max_epochs=1000)
    #r.weights = r.best_weights
    print r.errors
    #plt.matshow(r.daydream(50), cmap=plt.cm.gray_r)
