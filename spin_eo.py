import numpy as np
import random
import operator
import collections
import math
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def setup_ising(n=20):
    #n by n ising model
    config = np.ones(n)
    weights = collections.defaultdict(random.random)
    for x in np.nditer(config, op_flats=['readwrite']):
        if random.random() > 0.5:
            x[...] = -1
    return (config, weights)

def conf_energy(config, weights, n):
    local_energy = np.zeros(n-2)
    for x in xrange(1,n-1):
        local_energy[x-1] = weights[(x,x+1)]*config[x]*config[x+1] + weights[(x-1,x)]*config[x-1]*config[x]
    hamiltonian = local_energy.sum()
    return (hamiltonian, local_energy)

def argmax(ls):
    return max(enumerate(ls), key=operator.itemgetter(1))[0]

def get_kth_highest_arg(ls, k):
    return sorted(enumerate(ls), key=operator.itemgetter(1), reverse=True)[k][0]

def swap_state(energies, curr_s):
    pass

def optimize_spinglass(config, weights, steps=10000, disp=False):
    best_s = config
    best_energy = float("inf")
    total_energy = float("inf")
    curr_s = best_s.copy()
    for time in xrange(steps):
        if disp and time % (steps // 20) == 0:
            print "time: ", time
        total_energy, energies = conf_energy(curr_s, weights)
        if total_energy < best_energy:
            best_energy = total_energy
            best_s = curr_s
        curr_s = swap_state(energies, curr_s)
    return best_s, best_energy

if __name__ == "__main__":
    config, weights = setup_ising(n=50)
    for x in xrange(2):
        opt_config, score = optimize_tsp(config, weights, steps=50000, disp=True)
        plt.matshow(opt_config)
        plt.matshow(weights)
        plt.show()
