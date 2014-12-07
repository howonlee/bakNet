import numpy as np
import math
import random

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
activation = np.array([1,1,0,0,1,0,1,1,0,1,0,0,0,1,1,0])

def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))

def random_activation():
    act = np.zeros(16)
    for x in xrange(16):
        if random.random() > 0.5:
            act[x] = 1
    return act

def update(activation, weights, numIters=1):
    unit = random.randint(0,15)
    netinput = weights[unit].dot(activation)
    if (random.random() < sigmoid(netinput)):
        activation[unit] = 1
    else:
        activation[unit] = 0
    return activation

if __name__ == "__main__":
    #just need to visualize it, actually
    pass
