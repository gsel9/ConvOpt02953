import numpy as np 
import matplotlib.pyplot as plt 

from data import data_matrix, mask_matrix, eval_loss


def shrinkage_threshold(x, gamma):
    """
    Applies the shrinkage/soft-threshold operator given a vector and a threshold. 
    It's defined in the equation 1.5 in the paper of FISTA.
    
    Input:
        x: vector to apply the soft-threshold operator.
        gamma: threshold.
    Output:
        x_out: vector after applying the soft-threshold operator.
    """
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def singular_value_shrinkage(X, gamma):
    """
    Performs the called singular value shrinkage defined 
        in the equation 2.11 from the report.
    
    Input:
        X: matrix to apply the singular value shrinkage operator.
        gamma: threshold.
        newshape: (optional) if we need to reshape the input (e.g. a vector),
            we can pass a new shape.
    Output:
        X_out: matrix after applying the singular value shrinkage operator.
    """

    U, s, VT = np.linalg.svd(X, full_matrices=True)
    s_th = shrinkage_threshold(s, gamma)

    S = np.zeros_like(X)
    np.fill_diagonal(S, s_th)
    
    return (U @ S @ VT)


def fista(X, O, i_eval=10, gamma=1, t_k=1, L_hat=0.5, n_iter=2):
    """
    Performs the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) defined
        in the paper of FISTA adapted to the problem of matrix completion.
    
    Input:
        X: noisy matrix.
        A: matrix that encodes the known entries
        gamma: threshold for the singular value shrinkage operator.
        n_iter: Number of iteration that the algorithm executes.

    Output:
        Y_k: reconstructed matrix after n_iter iterations of the algorithm.
    """

    X_k = X.copy()
    Y_k = X.copy() 

    gamma = gamma / (1 / L_hat)

    solutions = []
    for i in range(n_iter):

        X_kk = singular_value_shrinkage(Y_k + L_hat * O * (X - Y_k), gamma)

        t_kk = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2

        Y_k = X_kk + ((t_k - 1) / t_kk) * (X_kk - X_k)

        X_k = X_kk
        t_k = t_kk

        if i % i_eval == 0:
            solutions.append(Y_k)
    
    solutions.append(Y_k)

    return solutions


def main_first_order():

    n_iter = 5000

    M = data_matrix()
    O = mask_matrix()
    X = M * O

    fig, ax = plt.subplots()

    for gamma in [0.5, 0.1, 0.01, 0.001]:
    
        solutions = fista(X, O, gamma=gamma, n_iter=n_iter)
        losses = [eval_loss(M, M_hat) for M_hat in solutions]

        print("Min loss:", losses[-1])

        ax.plot(losses, label=f"$\gamma = {{{gamma}}}$")
    
    ax.set_xticks(np.linspace(0, len(losses), 6, dtype=int))
    ax.set_xticklabels(np.linspace(1, n_iter, 6, dtype=int))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.axhline(y=0.3382029640059954, linestyle="--", c="maroon", 
               label=r"CVXPY ($\gamma = 0.01$)", alpha=0.7)

    plt.legend()
    plt.savefig("loss_approx.pdf")


if __name__ == "__main__":
    main_first_order()
