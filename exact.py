import numpy as np 
import cvxpy as cp

from data import data_matrix, mask_matrix, eval_loss


def nuclear_norm_solver(A, mask, gamma=1.0):
    """
    Solve using a nuclear norm approach, using CVXPY.
    [ Candes and Recht, 2009 ]
    Parameters:
    -----------
    A : m x n array
        matrix we want to complete
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)
    gamma : float
        hyperparameter controlling tradeoff between nuclear norm and square loss
    Returns:
    --------
    X: m x n array
        completed matrix
    """

    X = cp.Variable(shape=A.shape)

    objective = cp.Minimize(gamma * cp.norm(X, "nuc") + cp.sum_squares(cp.multiply(mask, X - A)))

    problem = cp.Problem(objective, [])
    problem.solve(solver=cp.SCS)

    return X.value 


def main_nuclear_norm():

    M = data_matrix()
    O = mask_matrix()
    X = M * O

    M_hat = nuclear_norm_solver(X, O, gamma=0.01)

    print(eval_loss(M, M_hat))


if __name__ == "__main__":
    main_nuclear_norm()
