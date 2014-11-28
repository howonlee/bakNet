import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def setup_tsp(n=20):
    #50 by 50, uniformly distributed in 2-space
    config = {}
    for label in xrange(n):
        config[label] = (random.random(), random.random())
    return config

def dist(tup1, tup2):
    #2-dimensional euclidean distance
    return math.sqrt((tup2[0] - tup1[0]) ** 2 + (tup2[1] - tup1[1]) ** 2)

def dist_matrix(config):
    config_ls = config.items()
    dim = len(config_ls)
    distmat = np.zeros((dim, dim))
    for first in config_ls:
        for second in config_ls:
            distmat[first[0], second[0]] = dist(first[1], second[1])
    return distmat

def get_random_solution(n):
    random_soln = range(n)
    random.shuffle(random_soln)
    return random_soln

def calc_city_energy(distmat, soln):
    energies = []
    for i, city in enumerate(soln):
        #e_i = p_i - min_{j \neq i} (d_{ij})
        j = soln[(i+1) % len(soln)]
        energies.append(distmat[city,j] - distmat[city,:].min())
    return energies

def calc_total_energy(distmat, city_energies):
    return sum(city_energies) + distmat.min(axis=0).sum()

def argmax(ls):
    return max(enumerate(ls), key=operator.itemgetter(1))[0]

def get_kth_highest_arg(ls, k):
    return sorted(enumerate(ls), key=operator.itemgetter(1), reverse=True)[k][0]

def swap_city(energies, soln, tau=1.5):
    k = len(soln)
    dist = len(soln)
    while k > len(soln)-1:
        k = int(np.random.pareto(tau))
    while dist > len(soln)-1:
        dist = int(np.random.pareto(tau))
    worst_city = get_kth_highest_arg(energies, k)
    new_soln = list(soln) #deep copy
    new_idx = (worst_city + dist) % len(soln)
    new_soln[new_idx], new_soln[worst_city] = new_soln[worst_city], new_soln[new_idx]
    return new_soln

def optimize_tsp(config, steps=10000, disp=False):
    best_s = get_random_solution(len(config))
    best_energy = float("inf")
    total_energy = float("inf")
    curr_s = list(best_s)
    distmat = dist_matrix(config)
    for time in xrange(steps):
        if disp and time % (steps // 20) == 0:
            print "time: ", time
        energies = calc_city_energy(distmat, curr_s)
        total_energy = calc_total_energy(distmat, energies)
        if total_energy < best_energy:
            best_energy = total_energy
            best_s = curr_s
        curr_s = swap_city(energies, curr_s)
    return best_s, best_energy

def show_tsp(config, order=None, save=False):
    config_ls = config.items()
    labels = map(operator.itemgetter(0), config_ls)
    xs = map(lambda x: x[1][0], config_ls)
    ys = map(lambda x: x[1][1], config_ls)
    plt.scatter(xs, ys, c=labels)
    if order:
        vertices = map(config.get, order)
        x, y = zip(*vertices)
        line, = plt.plot(x, y, 'go-')
    if save:
        pass
    else:
        plt.show()

def show_distmat(distmat):
    plt.matshow(distmat)
    plt.show()

if __name__ == "__main__":
    #show_tsp(config)
    config = setup_tsp(n=50)
    for x in xrange(2):
        order, score = optimize_tsp(config, steps=50000, disp=True)
        print order, score
        show_tsp(config, order)
