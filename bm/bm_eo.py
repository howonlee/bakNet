import numpy as np
import random
import operator
import itertools
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
    weights = np.random.rand(n,n) #will be sparse
    return (config, weights)

@jit
def conf_energy(config, weights):
    dims = config.shape
    local_energy = np.zeros_like(config)
    for x in xrange(0, dims[0]):
        for y in xrange(0, dims[0]):
            if x == y: continue
            local_energy[x] += weights[x,y] * config[x] * config[y]
    hamiltonian = local_energy.sum()
    return (hamiltonian, local_energy)

#just do the parity bits function, basically
def gen_paritybits(bits=5):
    bits_ls = [map(int, seq) + [sum(map(int, seq)) % 2] for seq in itertools.product("01", repeat=bits)]
    return bits_ls

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

def learn_bm(config, weights, pat):
    #optimize the bm without the clamp
    #optimize the bm with the clamp
    #what does it mean to take the equilibrium statistics of such a thing?
    #take the difference, assume T=1
    return (config, weights)

def optimize_bm(config, weights, steps=10000, disp=False, clamp=None):
    #clamp is simple Python array always
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
    parity_bits = gen_paritybits()
    for idx, x in enumerate(parity_bits):
        if idx % 3: #if you did % 2, you would be in big trouble
            config, weights = learn_bm(config, weights, x)
        else:
            pass
    #now try the performance by clamping and optimizing...
