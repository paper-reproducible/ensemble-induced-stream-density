import tensorflow as tf
from scipy import sparse


def coo_matrix(X: tf.Tensor, shape=None, dtype=None, copy=False):
    return sparse.coo_matrix(X.numpy(), shape, dtype, copy)

def hstack(blocks, format=None, dtype=None):
    return sparse.hstack(blocks, format, dtype)