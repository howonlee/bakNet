import numpy as np
import math
import random
import itertools
import datetime
import sys

weights = np.array([
    [0,2,0,2,2,0,0,0,-3,-3,0,0,0,0,0,0],
    [2,0,2,0,0,2,0,0,-3,-3,0,0,0,0,0,0],
    [0,2,0,2,0,0,2,0,0,0,-3,-3,0,0,0,0],
    [2,0,2,0,0,0,0,2,0,0,-3,-3,0,0,0,0],
    [2,0,0,0,0,2,0,2,0,0,0,0,-3,-3,0,0],
    [0,2,0,0,2,0,2,0,0,0,0,0,-3,-3,0,0],
    [0,0,2,0,0,2,0,2,0,0,0,0,0,0,-3,-3],
    [0,0,0,2,2,0,2,0,0,0,0,0,0,0,-3,-3],
    [-3,-3,0,0,0,0,0,0,0,2,0,2,2,0,0,0],
    [-3,-3,0,0,0,0,0,0,2,0,2,0,0,2,0,0],
    [0,0,-3,-3,0,0,0,0,0,2,0,2,0,0,2,0],
    [0,0,-3,-3,0,0,0,0,2,0,2,0,0,0,0,2],
    [0,0,0,0,-3,-3,0,0,2,0,0,0,0,2,0,2],
    [0,0,0,0,-3,-3,0,0,0,2,0,0,2,0,2,0],
    [0,0,0,0,0,0,-3,-3,0,0,2,0,0,2,0,2],
    [0,0,0,0,0,0,-3,-3,0,0,0,2,2,0,2,0]])

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def random_activation():
    act = np.ones(16)
    for x in xrange(16):
        if random.random() > 0.5:
            act[x] = -1
    return act

def noise(x):
    """
    Unused
    """
    rand_vec = np.random.random_sample(x.shape)
    rand_vec -= 0.5
    rand_vec *= 0.02
    return x + rand_vec

def update(activation, weights, tau=1.5):
    #get local energies
    energies = np.dot(weights, activation) * activation
    if tau < 0:
        unit = np.argmin(energies)
    else:
        k = 16
        while k > 15:
            k = int(np.random.pareto(tau))
        unit = energies.argsort()[k]
    activation[unit] = 1 - activation[unit]
    return activation

if __name__ == "__main__":
    global_minima = set()
    global_1 = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])
    global_2 = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    global_1.flags.writeable = False
    global_2.flags.writeable = False
    global_minima.add(hash(global_1.data))
    global_minima.add(hash(global_2.data))
    activations = reversed([np.array(map(int, seq)) for seq in itertools.product("01", repeat=16)])
    c = datetime.datetime.now()
    curr_time = c.second * 1000000 + c.microsecond
    print curr_time
    for activation in activations:
        i = 0
        activation_store = []
        while True:
            i += 1
            activation = update(activation, weights)
            activation_copy = activation.copy()
            activation_store.append(activation_copy)
            activation_copy.flags.writeable = False
            if hash(activation_copy.data) in global_minima:
                if i > 100:
                    for line in activation_store:
                        print line
                    sys.exit(0)
