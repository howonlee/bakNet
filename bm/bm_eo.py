import numpy as np
import random
import operator
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
            x[...] = -0
    weights = np.random.rand(n,n)
    weights = (weights + weight.T) / 2 #symmetry
    np.fill_diagonal(weights, 0) #no self-connections
    return (config, weights)

@jit
def draw_from_config(config, n=5):
    #configuration is actually a distribution
    #draw logistic
    draw = np.zeros_like(config)
    for x in xrange(n):
        for y in xrange(len(config)):
            if np.random.rand() < y: #this is wrong
                draw[x] += 1
    draw /= float(n)
    return draw

@jit
def conf_energy(config, weights):
    #in the BM case, this is actually the energy delta
    #\Delta E_k = \sum_i w_{ki} s_i
    dims = config.shape
    local_energy = np.zeros_like(config)
    #draw = draw_from_config(config)
    for x in xrange(0, dims[0]): #k
        for y in xrange(0, x):
            local_energy[x] += weights[x,y] * draw[y]
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
    worst = -1
    while worst > clamp:
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
    soln_wo_clamp, _ = optimize_bm(config, weights, steps=200)
    soln_w_clamp, _ = optimize_bm(config, weights, steps=200, clamp=pat)
    # I need p_ij's
    for x in xrange(len(soln_wo_clamp)):
        for y in xrange(len(soln_w_clamp)):
            weights[x,y] -= soln_wo_clamp[x] * soln_wo_clamp[y]
    return (config, weights)

def optimize_bm(config, weights, steps=10000, disp=False, clamp=None):
    #clamp is simple Python array always
    best_s = config
    best_energy = float("inf")
    total_energy = float("inf")
    curr_s = best_s.copy()
    for time in xrange(steps):
        total_energy, energies = conf_energy(curr_s, weights)
        print total_energy
        if total_energy < best_energy:
            best_energy = total_energy
            best_s = curr_s
        curr_s = flip_state(energies, curr_s, use_k=False, clamp=len(clamp))
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
