import numpy as np
import random
import operator
import math
import itertools
import collections
import math
from numba import jit #requires anaconda
import matplotlib.pyplot as plt

def setup_bm(n=100):
    #n lattice point bm
    #hidden nodes are implicit
    config = np.ones(n)
    for x in np.nditer(config, op_flags=['readwrite']):
        if random.random() > 0.5:
            x[...] = 0
    weights = np.random.rand(n,n) - 0.5
    weights = (weights + weights.T) / 2 #symmetry
    np.fill_diagonal(weights, 0) #no self-connections
    return (config, weights)

"""
I don't think I need this... do I?

def sigmoid(x):
    #assume temperature's 1
    return 1.0 / (1 + math.exp(-x))

@jit
def draw_from_config(energy_delta, n=1):
    #configuration is actually a distribution
    #draw logistic
    draw = np.zeros_like(energy_delta)
    for x in xrange(n):
        for y in energy_delta:
            if np.random.rand() < logistic(energy_delta): #this is wrong
                draw[x] += 1
    return draw
"""

@jit
def conf_energy(config, weights):
    #in the BM case, this is actually the energy delta
    #\Delta E_k = \sum_i w_{ki} s_i
    dims = config.shape
    local_energy = np.zeros_like(config)
    for x in xrange(0, dims[0]): #k
        for y in xrange(0, x):
            local_energy[x] -= weights[x,y] * config[x] * config[y]
    hamiltonian = local_energy.sum()
    return (hamiltonian, local_energy)

#just do the parity bits function, basically
def gen_paritybits(bits=5):
    bits_ls = [map(int, seq) + [sum(map(int, seq)) % 2] for seq in itertools.product("01", repeat=bits)]
    return bits_ls

def gen_bits(bits=5):
    return [map(int, seq) for seq in itertools.product("01", repeat=bits)]

def flip_state(energies, soln, tau=1.1, use_k=True, clamp=-1):
    #here, clamp is the index of the last clamp
    #### index via the ravel
    k = 0
    worst = -5
    if clamp:
        while worst < clamp:
            k = soln.size
            while k > soln.size-1:
                k = int(np.random.pareto(tau))
            worst = energies.ravel().argsort()[-(k+1)]
    else:
        if use_k:
            k = soln.size #eventually, the ravel solution
            while k > soln.size-1:
                k = int(np.random.pareto(tau))
        worst = energies.ravel().argsort()[-(k+1)]
    #worst = energies.ravel().argsort()[-(k+1)]
    new_soln = soln.copy()
    #it shouldn't be flipping, but I can't think of what...
    new_soln[worst] = 1 - new_soln[worst]
    return new_soln

def make_pij(conf):
    return np.outer(conf, conf)

def learn_bm(config, weights, pat):
    solns_wo_clamp = sample_bm(config, weights, steps=200)
    solns_w_clamp = sample_bm(config, weights, steps=200, clamp=pat)
    p_ij = make_pij(solns_wo_clamp)
    p_ij_prime = make_pij(solns_w_clamp)
    # I need p_ij's
    # make the p_ij matrices, basically
    weights -= (p_ij - p_ij_prime)
    np.fill_diagonal(weights, 0) #no self-connections
    return (config, weights)

def sample_bm(config, weights, steps=100, disp=False, clamp=None):
    #clamp is simple Python array always
    best_s = config
    if clamp:
        for idx, val in enumerate(clamp):
            best_s[idx] = val
    best_energy = float("inf")
    total_energy = float("inf")
    curr_s = best_s
    for time in xrange(steps):
        total_energy, energies = conf_energy(curr_s, weights)
        if total_energy < best_energy:
            best_energy = total_energy
            best_s = curr_s
        clamp_val = len(clamp) if clamp else -1
        curr_s = flip_state(energies, curr_s, use_k=False, clamp=clamp_val)
    return best_s

if __name__ == "__main__":
    config, weights = setup_bm(n=7)
    parity_bits = gen_paritybits()
    for idx, x in enumerate(parity_bits):
        config, weights = learn_bm(config, weights, x)
    plt.matshow(weights, cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()
    parity_bits_2 = gen_bits() #without the answer
    for bits in parity_bits_2:
        print sum(bits) % 2, sample_bm(config, weights, 1, clamp=bits)[-1]
    #now try the performance by clamping and optimizing...
