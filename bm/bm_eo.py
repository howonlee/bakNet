import numpy as np
import random
import operator
import collections
import math
from numba import jit #requires anaconda
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def setup_bm(n=100):
    #n lattice point bm
    #hidden nodes are implicit
    config = np.ones(n)
    for x in np.nditer(config, op_flags=['readwrite']):
        if random.random() > 0.5:
            x[...] = -1
    weights = np.random.rand((n,n)) #will be sparse
    return (config, weights)

@jit
def conf_energy(config, weights):
    dims = config.shape
    local_energy = np.zeros_like(config)
    for x in xrange(0, dims[0]):
        x_plus = (x + 1) % dims[0]
        x_minus = (x - 1) % dims[0]
        local_energy[x] += weights[x,x_plus] * config[x] * config[x_plus]
        local_energy[x] += weights[x_minus,x] * config[x_minus] * config[x]
    hamiltonian = local_energy.sum()
    return (hamiltonian, local_energy)

def swap_state(energies, soln, tau=1.1, use_k=True):
    #### index via the ravel
    k = 0
    if use_k:
        k = soln.size #eventually, the ravel solution
        while k > soln.size-1:
            k = int(np.random.pareto(tau))
    worst = energies.ravel().argsort()[-(k+1)]
    new_soln = soln.copy()
    new_idx = random.randint(0,soln.size-1)
    new_soln.flat[new_idx], new_soln.flat[worst] = new_soln.flat[worst], new_soln.flat[new_idx]
    return new_soln

def learn_bm(config, weights):
    pass

def optimize_bm(config, weights, steps=10000, disp=False):
    best_s = config
    best_energy = float("inf")
    total_energy = float("inf")
    curr_s = best_s.copy()
    for time in xrange(steps):
        total_energy, energies = conf_energy(curr_s)
        print total_energy
        if total_energy < best_energy:
            best_energy = total_energy
            best_s = curr_s
        curr_s = swap_state(energies, curr_s, use_k=False)
    return best_s, best_energy

if __name__ == "__main__":
    config, weights = setup_bm(n=30)
    for x in xrange(1):
        opt_config, score = optimize_spinglass(config, steps=4000, disp=True)
        config, weights = learn_bm(config, weights)
        #print opt_config.shape
        #plt.matshow(opt_config)
        #plt.show()
