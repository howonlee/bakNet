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

    def __init__(self, reg_lambda=0, epsilon_init=0.12, hidden_layer_size=25, maxiter=500):
        self.reg_lambda = reg_lambda
        self.epsilon_init = epsilon_init
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = self.sigmoid
        self.activation_func_prime = self.sigmoid_prime
        self.method = opti_method
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

    def _forward(self, X, t1, t2):
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

    def function(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

        m = X.shape[0]
        Y = np.eye(num_labels)[y]

        _, _, _, _, h = self._forward(X, t1, t2)
        costPositive = -Y * np.log(h).T
        costNegative = (1 - Y) * np.log(1 - h).T
        cost = costPositive - costNegative
        J = np.sum(cost) / m

        if reg_lambda != 0:
            t1f = t1[:, 1:]
            t2f = t2[:, 1:]
            reg = (self.reg_lambda / (2 * m)) * (self.sumsqr(t1f) + self.sumsqr(t2f))
            J = J + reg
        return J

    def eo(self, fn, thetas, args=None):
        if args == None:
            print "need the args"
            return None
        pass

    def fit(self, X, y):
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        try:
            num_labels = len(set(y))
        except:
            num_labels = y.size

        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
        theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
        thetas0 = self.pack_thetas(theta1_0, theta2_0)

        _res = self.eo(self.function, thetas0, args=(input_layer_size, self.hidden_layer_size, num_labels, X, y, 0))

        self.t1, self.t2 = self.unpack_thetas(_res.x, input_layer_size, self.hidden_layer_size, num_labels)

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
    nn.fit(X_train, y_train)
    print accuracy_score(y_test, nn.predict(X_test))

def iris_class():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)
    nn = EONet(hidden_layer_size=25)
    nn.fit(X_train, y_train)
    print accuracy_score(y_test, nn.predict(X_test))

if __name__ == "__main__":
    mnist_digits()
    #iris_class()
