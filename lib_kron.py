"""
Kronecker graph generation lib
"""
import numpy as np
import numpy.random as npr
import scipy.sparse as sci_sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import pymc

def create_generator(generator, exponent):
    """
    Create the dense kronecker distributional array
    """
    arr = generator.copy()
    for x in xrange(exponent):
        arr = np.kron(arr, generator)
    return arr

def generate(arr):
    k_mat = sci_sp.dok_matrix(arr.shape)
    for i in xrange(arr.shape[0]):
        for j in xrange(arr.shape[0]):
            rand = random.random()
            if rand < arr[i,j]:
                k_mat[i,j] = 1
    return k_mat

def evaluate_gradient(param_mat, graph_mat):
    """
    Could calculate the log-likelihood, too, actually
    """
    pass

def update_params(estimate, grad):
    pass

def kron_fit(mat, est_size=2):
    """
    Kronecker fit on not too big a space
    """
    mat_shape = mat.shape
    assert mat.shape[0] % est_size == 0
    estimate = npr.random((est_size,est_size))
    """
    while True:
        grad = evaluate_gradient()
        estimate = update_params(estimate, grad)
        if converged:
            break
    """
    return estimate

def kron_em():
    """
    Kronecker em
    To be implemented when I have the time
    """
    pass
