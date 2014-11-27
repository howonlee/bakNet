import numpy as np
import random
import operator
import matplotlib.pyplot as plt

def setup_tsp(n=20):
    #50 by 50, uniformly distributed in 2-space
    config = {}
    for label in xrange(n):
        config[label] = (random.random() * 50, random.random() * 50)
    return config

def dist_matrix(config):
    config_ls = config.items()
    dim = len(config_ls)
    distmat = np.zeros((dim, dim))
    for x in config_ls:
        for y in config_ls:
            distmat[x[0], y[0]] = dist ##### make the dist
    return distmat

def get_random_solution(n=20):
    random_soln = range(n)
    random.shuffle(random_soln)
    return random_soln

def calc_city_energy(distmat, soln):
    pass

def calc_total_energy(distmat, soln):
    pass

def select_city(energies, soln, alpha):
    pass

def optimize_tsp():
    pass

def show_tsp(config, order=None):
    config_ls = config.items()
    labels = map(operator.itemgetter(0), config_ls)
    xs = map(lambda x: x[1][0], config_ls)
    ys = map(lambda x: x[1][1], config_ls)
    plt.scatter(xs, ys, c=labels)
    plt.show()

if __name__ == "__main__":
    config = setup_tsp()
    show_tsp(config)
