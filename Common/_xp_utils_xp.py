# pyright: reportMissingImports=false
import numpy


def _unique(X, xp=numpy):
    if len(X.shape) == 2:
        X_ = xp.ascontiguousarray(X).view(
            xp.dtype((xp.void, X.dtype.itemsize * X.shape[1]))
        )
        _, idx = xp.unique(X_, return_index=True)
        X_ = X[idx]
    elif len(X.shape) == 1:
        X_ = xp.unique(X)
    return X_


def _tensor_scatter_nd_update(X, indices, updates):
    X_ = X.copy()
    X_[indices] = updates

    return X_


def setup_cupy(xpUtils):
    import cupy

    setattr(xpUtils, "coo_matrix", cupy.sparse.coo_matrix)
    setattr(xpUtils, "hstack", cupy.sparse.hstack)
    setattr(xpUtils, "norm", cupy.linalg.norm)
    setattr(xpUtils, "to_numpy", lambda X: X.get())
    setattr(xpUtils, "unique", lambda X: _unique(X, cupy))
    setattr(xpUtils, "copy", lambda X: X.copy())
    setattr(xpUtils, "tile", cupy.tile)
    setattr(xpUtils, "cast", lambda X, dtype: X.astype(dtype))
    setattr(xpUtils, "numpy_dtype", lambda X: X.dtype)
    setattr(xpUtils, "tensor_scatter_nd_update", _tensor_scatter_nd_update)

    return cupy


def setup_numpy(xpUtils):
    import scipy

    setattr(xpUtils, "coo_matrix", scipy.sparse.coo_matrix)
    setattr(xpUtils, "hstack", scipy.sparse.hstack)
    setattr(xpUtils, "norm", numpy.linalg.norm)
    setattr(xpUtils, "to_numpy", lambda X: X)
    setattr(xpUtils, "unique", lambda X: _unique(X, numpy))
    setattr(xpUtils, "copy", lambda X: X.copy())
    setattr(xpUtils, "tile", numpy.tile)
    setattr(xpUtils, "cast", lambda X, dtype: X.astype(dtype))
    setattr(xpUtils, "numpy_dtype", lambda X: X.dtype)
    setattr(xpUtils, "tensor_scatter_nd_update", _tensor_scatter_nd_update)

    return numpy
