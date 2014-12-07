import numpy as np
import random
import operator
import math
import itertools
import collections
import math
from numba import jit #requires anaconda
import matplotlib.pyplot as plt

"""
We need to be as sure about everything here as possible before we also implement the simulated annealing
"""

def rand_vec(n=100):
    vec = np.ones(n)
    for x in np.nditer(vec, op_flags=['readwrite']):
        if random.random() > 0.5:
            x[...] = 0
    return vec

def setup_bm(n=100):
    #n lattice point bm
    #hidden nodes are implicit
    config = rand_vec(n)
    weights = np.random.rand(n,n) - 0.5
    weights = (weights + weights.T) / 2 #symmetry
    np.fill_diagonal(weights, 0) #no self-connections
    return (config, weights)

@jit
def conf_energy(config, weights):
    #in the BM case, this is actually the energy delta
    #\Delta E_k = \sum_i w_{ki} s_i
    dims = config.shape
    local_energy = np.zeros_like(config)
    for x in xrange(0, dims[0]): #k
        for y in xrange(0, x):
            local_energy[x] -= weights[x,y] * config[y] #?
    hamiltonian = local_energy.sum()
    return (hamiltonian, local_energy)

#just do the parity bits function, basically
def gen_paritybits(bits=2):
    bits_ls = [map(int, seq) + [sum(map(int, seq)) % 2] for seq in itertools.product("01", repeat=bits)]
    return bits_ls

def gen_bits(bits=2):
    return [map(int, seq) for seq in itertools.product("01", repeat=bits)]


def flip_state(energies, soln, tau=1.1, use_k=True, clamp=-1):
    #here, clamp is the index of the last clamp
    #### index via the ravel
    k = 0
    worst = -5
    if clamp > 0:
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
    new_soln = soln.copy()
    #it shouldn't be flipping, but I can't think of what...
    new_soln[worst] = 1 - new_soln[worst]
    return new_soln

def learn_bm(config, weights, pat):
    wo_clamp = sample_bm(config, weights, steps=500)
    w_clamp = sample_bm(config, weights, steps=500, clamp=pat)
    p_ij= np.outer(w_clamp, w_clamp)
    p_ij_prime = np.outer(wo_clamp, wo_clamp)
    # I need p_ij's
    # make the p_ij matrices, basically
    weights -= (p_ij - p_ij_prime)
    #np.fill_diagonal(weights, 0) #no self-connections
    return (config, weights)

def sample_bm(config, weights, steps=100, disp=False, clamp=None):
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
    n = 4
    config, weights = setup_bm(n=n)
    parity_bits = gen_paritybits()
    for i in xrange(15):
        for idx, x in enumerate(parity_bits):
            config, weights = learn_bm(rand_vec(n), weights, x)
    plt.matshow(weights, cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()
    parity_bits_2 = gen_bits() #without the answer
    for bits in parity_bits_2:
        print sum(bits) % 2, sample_bm(config, weights, 10, clamp=bits)
    #now try the performance by clamping and optimizing...
