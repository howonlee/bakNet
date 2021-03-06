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

def sigmoid(x, t):
    return 1.0 / (1 + math.exp(-(x / t)))

def random_activation():
    act = np.zeros(16)
    for x in xrange(16):
        if random.random() > 0.5:
            act[x] = 1
    return act

def update(activation, weights, t):
    unit = random.randint(0,15)
    netinput = weights[unit].dot(activation)
    if (random.random() < sigmoid(netinput, t)): #t should come in here somehow
        activation[unit] = 1
    else:
        activation[unit] = 0
    return activation

if __name__ == "__main__":
    global_minima = set()
    global_1 = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])
    global_2 = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    global_1.flags.writeable = False
    global_2.flags.writeable = False
    global_minima.add(hash(global_1.data))
    global_minima.add(hash(global_2.data))
    activations = [np.array(map(int, seq)) for seq in itertools.product("01", repeat=16)]
    schedule = np.linspace(2,0.50,num=20)
    for activation in activations:
        i = 0
        while True:
            curr_temp = 0.5
            if i < schedule.size:
                curr_temp = schedule[i]
            activation = update(activation, weights, curr_temp)
            activation_copy = activation.copy()
            activation_copy.flags.writeable = False
            i += 1
            if hash(activation_copy.data) in global_minima:
                print i
                break
