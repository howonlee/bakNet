import numpy as np
import math
import random
import itertools

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
    return 1.0 / (1 + math.exp(-x))

def random_activation():
    act = np.zeros(16)
    for x in xrange(16):
        if random.random() > 0.5:
            act[x] = 1
    return act

def update(activation, weights, tau=1.5):
    #get local energies
    energies = np.zeros(16)
    for i in xrange(16):
        for j in xrange(16):
            energies[i] += weights[i][j] * activation[j] * activation[i]
    #print energies
    unit = np.argmin(energies)
    activation[unit] = 1 - activation[unit]
    return activation

if __name__ == "__main__":
    activations = [np.array(map(int, seq)) for seq in itertools.product("01", repeat=16)]
    for activation in activations:
        print "================"
        print activation
        for i in xrange(100):
            activation = update(activation, weights)
            print activation
