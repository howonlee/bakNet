import numpy as np
import random
import operator
import collections
import math
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def setup_ising(n=20):
    #n by n ising model
    config = np.ones((n,n))
    for x in np.nditer(config, op_flags=['readwrite']):
        if random.random() > 0.5:
            x[...] = -1
    return config

@jit
def conf_energy(config):
    dims = config.shape
    local_energy = np.zeros_like(config)
    for x in xrange(0, dims[0]):
        for y in xrange(0, dims[1]):
            if (x+1 < dims[0]):
                local_energy[x, y] += config[x,y] * config[x+1,y]
            if (x-1 > 0):
                local_energy[x, y] += config[x-1,y] * config[x,y]
            if (y+1 < dims[1]):
                local_energy[x, y] += config[x,y] * config[x,y+1]
            if (y-1 > 0):
                local_energy[x, y] += config[x,y-1] * config[x,y]
    hamiltonian = local_energy.sum()
    return (hamiltonian, local_energy)

def swap_state(energies, soln, tau=1.1):
    #### index via the ravel
    k = soln.size #eventually, the ravel solution
    while k > soln.size-1:
        k = int(np.random.pareto(tau))
    worst = energies.ravel().argsort()[-(k+1)]
    new_soln = soln.copy()
    new_idx = random.randint(0,soln.size-1)
    new_soln.flat[new_idx], new_soln.flat[worst] = new_soln.flat[worst], new_soln.flat[new_idx]
    return new_soln

def optimize_spinglass(config, steps=10000, disp=False):
    best_s = config
    best_energy = float("inf")
    total_energy = float("inf")
    curr_s = best_s.copy()
    for time in xrange(steps):
        if disp and time % (steps // 200) == 0:
            #print "time: ", time
            print best_energy
        total_energy, energies = conf_energy(curr_s)
        if total_energy < best_energy:
            best_energy = total_energy
            best_s = curr_s
        curr_s = swap_state(energies, curr_s)
    return best_s, best_energy

if __name__ == "__main__":
    config = setup_ising(n=60)
    for x in xrange(1):
        opt_config, score = optimize_spinglass(config, steps=40000, disp=True)
        #print opt_config.shape
        plt.matshow(opt_config)
        plt.show()
