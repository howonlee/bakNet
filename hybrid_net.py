import sys
import numpy as np
import numpy.random as npr
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

def tanh_deriv(x):
    return 1.0 - x**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_deriv(x):
    return logistic(x) * (1-logistic(x))

class BakBPNet(object):
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = np.tanh
            self.activation_deriv = tanh_deriv
        self.weights = []
        for i in range(1,len(layers)-1):
            self.weights.append((2*np.random.random((layers[i-1] + 1, layers[i]+ 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i+1]))-1)*0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000, extremal=True):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

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

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "xor":
        xor_prob()
    else:
        digits = load_digits() #from sklearn
        X = digits.data
        y = digits.target
        X -= X.min()
        X /= X.max()
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
