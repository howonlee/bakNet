from __future__ import division
import numpy as np
from scipy import optimize
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
import gzip
import cPickle

class EONet(object):

    def __init__(self, hidden_layer_size=25, maxiter=20000):
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = self.sigmoid
        self.activation_func_prime = self.sigmoid_prime
        self.maxiter = maxiter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def sumsqr(self, a):
        return np.sum(a ** 2)

    def rand_init(self, l_in, l_out):
        return np.random.rand(l_out, l_in + 1) * 2 * self.epsilon_init - self.epsilon_init

    def pack_thetas(self, t1, t2):
        return np.concatenate((t1.reshape(-1), t2.reshape(-1)))

    def unpack_thetas(self, thetas, input_layer_size, hidden_layer_size, num_labels):
        t1_start = 0
        t1_end = hidden_layer_size * (input_layer_size + 1)
        t1 = thetas[t1_start:t1_end].reshape((hidden_layer_size, input_layer_size + 1))
        t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))
        return t1, t2

    def argmax(self, ls):
        return max(enumerate(ls), key=operator.itemgetter(1))[0]

    def get_kth_highest_arg(self, ls, k):
        return sorted(enumerate(ls), key=operator.itemgetter(1), reverse=True)[k][0]

    def swap_city(self, energies, soln, tau=1.15):
        ################################
        k = len(soln)
        while k > len(soln)-1:
            k = int(np.random.pareto(tau))
        worst_city = get_kth_highest_arg(energies, k)
        new_soln = list(soln) #deep copy
        rand_idx = random.randrange(0, len(new_soln))
        new_soln[rand_idx], new_soln[worst_city] = new_soln[worst_city], new_soln[rand_idx]
        return new_soln

    def _forward(self, X, t1, t2):
        """
        Unchanged from the feedforward architecture
        """
        m = X.shape[0]
        ones = None
        if len(X.shape) == 1:
            ones = np.array(1).reshape(1,)
        else:
            ones = np.ones(m).reshape(m,1)

        # Input layer
        a1 = np.hstack((ones, X))

        # Hidden Layer
        z2 = np.dot(t1, a1.T)
        a2 = self.activation_func(z2)
        a2 = np.hstack((ones, a2.T))

        # Output layer
        z3 = np.dot(t2, a2.T)
        a3 = self.activation_func(z3)
        return a1, z2, a2, z3, a3

    def calc_local_energy(self, distmat, soln):
        ######################################
        energies = []
        for i, city in enumerate(soln):
            #e_i = p_i - min_{j \neq i} (d_{ij})
            j = soln[(i+1) % len(soln)]
            energies.append(distmat[city,j] - distmat[city,:].min())
        return energies

    def calc_total_energy(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y):
        ################################
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

        m = X.shape[0]
        Y = np.eye(num_labels)[y]

        _, _, _, _, h = self._forward(X, t1, t2)
        costPositive = -Y * np.log(h).T
        costNegative = (1 - Y) * np.log(1 - h).T
        cost = costPositive - costNegative
        return np.sum(cost) / m #J

    def eo(self, X, y):
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        try:
            num_labels = len(set(y))
        except:
            num_labels = y.size

        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
        theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
        thetas0 = self.pack_thetas(theta1_0, theta2_0)
        #######################
        best_s = get_random_solution(len(config))
        best_energy = float("inf")
        total_energy = float("inf")
        curr_s = list(best_s)
        distmat = dist_matrix(config)
        for time in xrange(steps):
            if disp and time % (steps // 20) == 0:
                print "time: ", time
            energies = calc_city_energy(distmat, curr_s)
            total_energy = calc_total_energy(distmat, energies)
            if total_energy < best_energy:
                best_energy = total_energy
                best_s = curr_s
            curr_s = swap_city(energies, curr_s)
        self.t1, self.t2 = self.unpack_thetas(best_s, input_layer_size, self.hidden_layer_size, num_labels)

    def predict(self, X):
        return self.predict_proba(X).argmax(0)

    def predict_proba(self, X):
        _, _, _, _, h = self._forward(X, self.t1, self.t2)
        return h

def mnist_digits():
    from scipy.io import loadmat
    data = loadmat('ex3data1.mat')
    X, y = data['X'], data['y']
    y = y.reshape(X.shape[0], )
    y = y - 1
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)
    nn = EONet(maxiter=100)
    nn.eo(X_train, y_train)
    print accuracy_score(y_test, nn.predict(X_test))

def iris_class():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)
    nn = EONet(hidden_layer_size=25)
    nn.eo(X_train, y_train)
    print accuracy_score(y_test, nn.predict(X_test))

if __name__ == "__main__":
    #mnist_digits()
    iris_class()
