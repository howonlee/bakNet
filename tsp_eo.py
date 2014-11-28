import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt

def setup_tsp(n=20):
    #50 by 50, uniformly distributed in 2-space
    config = {}
    for label in xrange(n):
        config[label] = (random.random() * 50, random.random() * 50)
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

def get_random_solution(n=20):
    random_soln = range(n)
    random.shuffle(random_soln)
    return random_soln

def calc_city_energy(distmat, soln):
    pass

def calc_total_energy(distmat, soln):
    pass

def swap_city(energies, soln, tau=1.15):
    pass

def optimize_tsp(steps=30000, n=20):
    config = setup_tsp(n)
    best_s = get_random_solution(n)
    best_energy = float("inf")
    curr_s = best_s
    distmat = distance_matrix(config)
    for time in xrange(steps):
        total_energy = calc_total_energy(distmat, curr_s)
        if total_energy < best_energy:
            best_energy = total_energy
            best_s = curr_s
        energies = calc_city_energy(distmat, curr_s)
        curr_s = swap_city(energies, curr_s)
    return best_s, best_energy

def show_tsp(config, order=None):
    config_ls = config.items()
    labels = map(operator.itemgetter(0), config_ls)
    xs = map(lambda x: x[1][0], config_ls)
    ys = map(lambda x: x[1][1], config_ls)
    plt.scatter(xs, ys, c=labels)
    plt.show()

def show_distmat(distmat):
    plt.matshow(distmat)
    plt.show()

if __name__ == "__main__":
    #show_tsp(config)
    show_distmat(dist_matrix(config))
    #print optimize_tsp()
