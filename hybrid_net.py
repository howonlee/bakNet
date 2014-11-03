import sys
import itertools
import cPickle
import gzip
import numpy as np
import numpy.random as npr
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

def normalize(mat):
    mat -= mat.min()
    mat /= mat.max()
    return mat

class BakBPNet(object):
    def __init__(self, layers):
        self.activation = np.tanh
        self.activation_deriv = lambda x: 1.0 - x**2
        self.weights = []
        for i in range(1,len(layers)-1):
            #this assumes a 1 layer net, because of the dimension assymetry
            #there's th(n+1) weight matrices, not th(2n)
            self.weights.append(self.scale_weights(npr.random((layers[i-1]+1, layers[i]+1))))
            self.weights.append(self.scale_weights(npr.random((layers[i]+1, layers[i+1]))))

    def scale_weights(self, wgts):
        return (2 * wgts - 1) * 0.25

    def add_bias(self, X):
        X = np.atleast_2d(X)
        temp = np.atleast_2d(np.ones(X.shape[0])).T
        return np.concatenate((X, temp),axis=1)

    def fit(self, X, y, learning_rate=0.2, epochs=10000, extremal=True):
        """
        SGD fit
        extremal = option for extremal adaptive dynamics
        """
        X = self.add_bias(X)
        y = np.array(y)
        epoch_frac = epochs // 20

        for k in range(epochs):
            if k % epoch_frac == 0:
                print "curr epoch is: ", k
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                activation = self.activation(np.dot(a[l], self.weights[l]))
                a.append(activation)
                if extremal:
                    max_hid_idxs = [np.argmax(activation)] # will be revised for deep net
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]
            if extremal:
                max_out_idx = np.argmax(self.weights[1,:]) #is this it?
                print max_hid_idxs, max_out_idx
                if y[max_out_idx] == 1: #is this it?
                    pass
                else:
                    self.hid_wgts[0][max_hid_idxs[0], :] -= curr_delta #revise
                    self.hid_wgts[0][min_hid_idxs[0], :] += (curr_delta * 0.8) #revise
                    self.out_wgt[max_out_idx, max_hid_idxs[-1]] -= curr_delta #revise
                    self.out_wgt[min_out_idx, min_hid_idxs[-1]] += (curr_delta * 0.8) #revise

            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

def xor_prob():
    nn = BakBPNet([2,2,1])
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    nn.fit(X, y)
    for i in [[0,0], [0,1], [1,0], [1,1]]:
        print i, nn.predict(i)

def sklearn_digits():
    digits = load_digits() #from sklearn
    X = normalize(digits.data)
    y = digits.target
    nn = BakBPNet([64,100,10])
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    nn.fit(X_train,labels_train,epochs=30000)
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i] )
        predictions.append(np.argmax(o))
    print confusion_matrix(y_test,predictions)
    print classification_report(y_test,predictions)

def mnist_digits():
    with gzip.open("mnist.pkl.gz", "rb") as f:
        train_set, valid_set, test_set = cPickle.load(f)
    X_train, y_train = train_set
    X_test, y_test = test_set
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
    nn = BakBPNet([784, 10000, 10])
    nn.fit(X_train, labels_train, epochs=50000)
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i])
        predictions.append(np.argmax(o))
    print confusion_matrix(y_test,predictions)
    print classification_report(y_test,predictions)

def parity_problem(bits):
    #stuff here
    X = np.array([map(int, seq) for seq in itertools.product("01", repeat=bits)])
    y = np.array([int(sum(x) % 2 == 0) for x in X])
    labels = LabelBinarizer().fit_transform(y)
    nn = BakBPNet([bits, 50, 2])
    nn.fit(X, labels)
    predictions = []
    for i in range(X.shape[0]):
        o = nn.predict(X[i])
        predictions.append(np.argmax(o))
    print confusion_matrix(y,predictions)
    print classification_report(y,predictions)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "xor":
        xor_prob()
    elif len(sys.argv) > 1 and sys.argv[1] == "sklearn":
        sklearn_digits()
    elif len(sys.argv) > 1 and sys.argv[1] == "mnist":
        mnist_digits()
    else:
        parity_problem(8)
