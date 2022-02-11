from ._utils import get_array_module
import numpy as np


# def zero_one_scale(X):
#     xp, _ = get_array_module(X)
#     X = X - xp.min(X, axis=0, keepdims=True)
#     X = X / xp.max(X, axis=0, keepdims=True)
#     return X


def random_rot(dim, dtype, xp=np):
    """Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(n)).
    The algorithm is described in the paper
    Stewart, G.W., "The efficient generation of random orthogonal
    matrices with an application to condition estimators", SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization"""
    H = xp.eye(dim, dtype=dtype)
    D = xp.ones((dim,), dtype=dtype)
    for n in range(1, dim):
        x = xp.random.normal(size=(dim - n + 1,)).astype(dtype)
        D[n - 1] = xp.sign(x[0])
        x[0] -= D[n - 1] * xp.sqrt((x * x).sum())
        # Householder transformation
        Hx = xp.eye(dim - n + 1, dtype=dtype) - 2.0 * xp.outer(x, x) / (x * x).sum()
        mat = xp.eye(dim, dtype=dtype)
        mat[n - 1 :, n - 1 :] = Hx
        H = xp.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - dim % 2) * D.prod()
    # Equivalent to mult(numx.diag(D), H) but faster
    H = (D * H.T).T
    return H


def rotate(X, SO=None):
    xp, _ = get_array_module(X)
    dims = X.shape[1]

    if SO is None:
        SO = random_rot(dims, dtype=X.dtype, xp=xp)

    return xp.matmul(X, SO), SO


def ball_scale(X):
    xp, _ = get_array_module(X)

    # move to zero
    center = (xp.min(X, axis=0, keepdims=True) + xp.max(X, axis=0, keepdims=True)) / 2
    X = X - center

    # scale
    d = xp.max(xp.linalg.norm(X, ord=2, axis=1))
    X = X / d

    return X


# See method 22 of http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
def ball_samples(psi, dims, xp=np):
    u = xp.reshape(xp.random.normal(0, 1, (dims + 2) * psi), [psi, dims + 2])
    norm = xp.linalg.norm(u, axis=1, keepdims=True)
    u = u / norm
    return u[:, 0:dims]


# if __name__ == "__main__":

#     import matplotlib.pyplot as plt

#     def scatter(l_x, l_y, l_label=None, show=True, block=True, fig=None, ax=None):
#         if fig is None:
#             fig = plt.figure()
#         if ax is None:
#             ax = plt.axes()
#             ax.set_xlim([0, 1])
#             ax.set_ylim([0, 1])
#         if l_label is not None:
#             ax.scatter(l_x, l_y, s=0.5, c=l_label)
#         else:
#             ax.scatter(l_x, l_y, s=0.5, c="blue")
#         if show:
#             plt.show(block=block)
#         return fig, ax

#     n = 2000
#     X = (ball_samples(n, 2, dtype=np.float32) + 1) / 2

#     scatter(X[:, 0], X[:, 1])
