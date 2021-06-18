import numpy as np 
import matplotlib.pyplot as plt 

from scipy.stats import bernoulli


def eval_loss(M, M_hat):

    return np.linalg.norm(M - M_hat) ** 2 / np.linalg.norm(M) ** 2


def mask_matrix(m=200, n=20, prob_masked=0.5, seed=42):
    """
    Generate a binary mask for m users and n movies.
    Note that 1 denotes observed, and 0 denotes unobserved.
    """
    np.random.seed(seed)

    return 1 - bernoulli.rvs(p=prob_masked, size=(m, n))


def data_matrix(m=200, n=20, k=15, seed=42):
    """
    Generate non-noisy data for m users and n movies with k latent factors.
    Draws factors U, V from Gaussian noise and returns U Vᵀ.
    """

    np.random.seed(seed)
    
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    
    return np.dot(U, V.T)
