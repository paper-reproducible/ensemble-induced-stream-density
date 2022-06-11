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
    if len(indices.shape) == 1:
        X_[indices] = updates
    elif len(indices.shape) == 2:
        X_[tuple([indices[:, i] for i in range(indices.shape[1])])] = updates
    return X_


def _setup_xp(xpUtils, xp):
    setattr(xpUtils, "unique", lambda X: _unique(X, xp))
    setattr(xpUtils, "copy", lambda X: X.copy())
    setattr(xpUtils, "tile", xp.tile)
    setattr(xpUtils, "cast", lambda X, dtype: X.astype(dtype))
    setattr(xpUtils, "numpy_dtype", lambda X: X.dtype)
    setattr(xpUtils, "tensor_scatter_nd_update", _tensor_scatter_nd_update)
    setattr(
        xpUtils,
        "gather_nd",
        lambda X, indices: X[tuple([indices[:, i] for i in range(indices.shape[1])])],
    )
    return xp


def setup_cupy(xpUtils):
    import cupy

    setattr(xpUtils, "coo_matrix", cupy.sparse.coo_matrix)
    setattr(xpUtils, "hstack", cupy.sparse.hstack)
    setattr(xpUtils, "norm", cupy.linalg.norm)
    setattr(xpUtils, "to_numpy", lambda X: X.get())
    return _setup_xp(xpUtils, cupy)


def setup_numpy(xpUtils):
    import scipy

    setattr(xpUtils, "coo_matrix", scipy.sparse.coo_matrix)
    setattr(xpUtils, "hstack", scipy.sparse.hstack)
    setattr(xpUtils, "norm", numpy.linalg.norm)
    setattr(xpUtils, "to_numpy", lambda X: X)
    return _setup_xp(xpUtils, numpy)
