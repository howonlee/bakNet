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
    weights = collections.defaultdict(random.random)
    for x in np.nditer(config, op_flags=['readwrite']):
        if random.random() > 0.5:
            x[...] = -1
    return (config, weights)

@jit
def conf_energy(config, weights):
    dims = config.shape
    local_energy = np.zeros((dims[0]-2, dims[1]-2))
    for x in xrange(1,dims[0]-1):
        for y in xrange(1,dims[1]-1): #4 local energies adjacent now...
            local_energy[x-1, y-1] = weights[(x,x+1,y,y)] * config[x,y] * config[x+1,y]
            local_energy[x-1, y-1] += weights[(x-1,x,y,y)] * config[x-1,y] * config[x,y]
            local_energy[x-1, y-1] += weights[(x,x,y,y+1)] * config[x,y] * config[x,y+1]
            local_energy[x-1, y-1] += weights[(x,x,y-1,y)] * config[x,y-1] * config[x,y]
    hamiltonian = local_energy.sum()
    return (hamiltonian, local_energy)

def swap_state(energies, soln, tau=1.5):
    #### index via the ravel
    k = soln.size #eventually, the ravel solution
    while k > soln.size-1:
        k = int(np.random.pareto(tau))
    worst = energies.ravel().argsort()[-(k+1)]
    new_soln = soln.copy()
    new_idx = random.randint(0,soln.size-1)
    new_soln.flat[new_idx], new_soln.flat[worst] = new_soln.flat[worst], new_soln.flat[new_idx]
    return new_soln

def optimize_spinglass(config, weights, steps=10000, disp=False):
    best_s = config
    best_energy = float("inf")
    total_energy = float("inf")
    curr_s = best_s.copy()
    for time in xrange(steps):
        if disp and time % (steps // 200) == 0:
            print "time: ", time
        total_energy, energies = conf_energy(curr_s, weights)
        if total_energy < best_energy:
            best_energy = total_energy
            best_s = curr_s
        curr_s = swap_state(energies, curr_s)
    return best_s, best_energy

if __name__ == "__main__":
    config, weights = setup_ising(n=50)
    for x in xrange(1):
        opt_config, score = optimize_spinglass(config, weights, steps=4000, disp=True)
        #print opt_config.shape
        plt.matshow(opt_config)
        plt.show()
