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
        self.activation_deriv = lambda x: 1.0 - np.tanh(x)**2
        self.weights = []
        for i in range(1,len(layers)-1):
            #this assumes a 1 layer net, because of the dimension assymetry
            #there's th(n+1) weight matrices, not th(2n)
            self.weights.append(self.scale_weights(npr.random((layers[i-1]+1, layers[i]+1))))
            self.weights.append(self.scale_weights(npr.random((layers[i]+1, layers[i+1]))))

    def scale_weights(self, wgts):
        return (2 * wgts - 1) * 0.25

    def add_bias(self, X):
        #add as in to concatenate, not to sum
        X = np.atleast_2d(X)
        temp = np.atleast_2d(np.ones(X.shape[0])).T
        return np.concatenate((X, temp),axis=1)

    def mse(self, err):
        return np.mean(np.square(err))

    def fit(self, X, y, learning_rate=0.2, epochs=10000, extremal=False):
        """
        SGD fit
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
            error = a[-1] - y[i]
            deltas = [error * self.activation_deriv(a[-1])]
            for l in range(len(a)-2, 0, -1): # we need to begin at the second to last layer
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
    print nn.weights[0]
    print nn.weights[1]
    for i in [[0,0], [0,1], [1,0], [1,1]]:
        print i, nn.predict(i)

def sklearn_digits(load=False):
    digits = load_digits() #from sklearn
    X = normalize(digits.data)
    y = digits.target
    nn = BakBPNet([64,100,10])
    if load:
        nn.weights[0] = np.load("bak_sk_hid_wgts.npy")[0].T
        nn.weights[1] = np.load("bak_sk_out_wgts.npy").T
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

def mnist_digits(load=False):
    with gzip.open("mnist.pkl.gz", "rb") as f:
        train_set, valid_set, test_set = cPickle.load(f)
    X_train, y_train = train_set
    X_test, y_test = test_set
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
    nn = BakBPNet([784, 2000, 10])
    if load:
        nn.weights[0] = np.load("bak_hid_wgts.npy")[0].T
        nn.weights[1] = np.load("bak_out_wgts.npy").T
    nn.fit(X_train, labels_train, epochs=20000)
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
    elif len(sys.argv) > 1 and sys.argv[1] == "mnist_preload":
        mnist_digits(load=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "sklearn_preload":
        sklearn_digits(load=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "parity":
        parity_problem(8)
    else:
        xor_prob()
