import numpy
from scipy import sparse as sci_sparse


def get_array_module(X):
    try:
        from cupy import get_array_module as cu_get_array_module
        from cupy import sparse as cu_sparse

        xp = cu_get_array_module(X)
        if xp == numpy:
            return xp, sci_sparse
        else:
            return xp, cu_sparse
    except:
        pass

    return numpy, sci_sparse


def to_numpy(X):
    try:
        from cupy import get_array_module

        xp = get_array_module(X)
        if xp == numpy:
            return X
        else:
            return X.get()
    except:
        pass

    return X


def asscalar(x):
    xp, _ = get_array_module(x)
    if "asscalar" in dir(xp):
        return xp.asscalar(x)
    elif "get" in dir(x):
        return numpy.asscalar(x.get())
    elif "numpy" in dir(x):
        return numpy.asscalar(x.numpy())


def set_printoptions():
    numpy.set_printoptions(formatter={"float_kind": "{:.4f}".format})


def unique(X):
    xp, _ = get_array_module(X)
    if len(X.shape) != 2:
        raise ValueError("Input array must be 2D.")
    sortarr = X[xp.lexsort(X.T[::-1])]
    mask = xp.empty(X.shape[0], dtype=xp.bool)
    mask[0] = True
    mask[1:] = xp.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]
