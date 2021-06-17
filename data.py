import numpy as np 
import matplotlib.pyplot as plt 

from scipy.stats import bernoulli


def eval_loss(M, M_hat):

    return np.linalg.norm(M - M_hat) ** 2 / np.linalg.norm(M) ** 2


def lowrank_missing(A, p, k, noise=0.0):
    """
    Generates an image A' where we have the rank k and missing entries. The
        given entries have optional noise.
    
    Input:
        A: the image we wish to corrupt.
        p: the probability that a given entry is missing in the final image.
        k: the rank of the final corrupted image.
        noise: the additive noise factor for our given entries. We assume that 
                our noise satisfies the normal distribution N(0,noise).
    Output:
        mask_rec: matrix of the corrupted image.
        mask: matrix with boolean values that indicates if the correspondent 
              pixel is known in the corrupted image.
    """
    m, n = A.shape
    mask = np.random.choice(np.arange(0, 2), p=[p, 1-p], size=(m,n))
    noise_matrix = np.random.normal(loc=0, scale = noise, size=(m,n))
    A_masked = np.multiply(A + noise_matrix,mask)
    U, s, VT = np.linalg.svd(A_masked, full_matrices=True )
    r = np.linalg.matrix_rank(A_masked)
    min_dim = min(m, n)
    s[min_dim - k:] = 0
    S = np.zeros((m, n), dtype=np.float)
    S[:min_dim, :min_dim] = np.diag(s)

    
    return U @ S @ VT, mask 


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
