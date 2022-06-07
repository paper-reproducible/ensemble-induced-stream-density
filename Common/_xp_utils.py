# pyright: reportMissingImports=false
import importlib
import importlib.util
import sys
from matplotlib.pyplot import axis

import numpy


_XPUTILS = "xp-utils"
_NP = "numpy"
_CUPY = "cupy"
_TNP = "tf.numpy"


def _get_array_module_name(X):
    try:
        import cupy

        xp = cupy.get_array_module(X)
        if xp == cupy:
            return _CUPY
    except:
        pass

    try:
        import tensorflow as tf

        if isinstance(X, tf.Tensor) and "numpy" in dir(X):
            return _TNP
    except:
        pass

    return _NP


def _unique(X, xp):
    X_ = xp.ascontiguousarray(X).view(
        xp.dtype((xp.void, X.dtype.itemsize * X.shape[1]))
    )
    _, idx = xp.unique(X_, return_index=True)

    return X[idx]


def _unique_tf(X, tf):
    X_, _ = tf.raw_ops.UniqueV2(x=X, axis=[0])
    return X_


def get_array_module_with_utils(arrayModuleName):
    utilsModuleName = _XPUTILS + "." + arrayModuleName
    if utilsModuleName in sys.modules:
        return sys.modules[utilsModuleName]
    spec = importlib.machinery.ModuleSpec(utilsModuleName, None)
    xp = numpy
    xpUtils = importlib.util.module_from_spec(spec)
    if arrayModuleName == _TNP:
        import tensorflow as tf
        from ._tf_sparse import coo_matrix, hstack

        setattr(xpUtils, "coo_matrix", coo_matrix)
        setattr(xpUtils, "hstack", hstack)
        setattr(xpUtils, "norm", tf.norm)
        setattr(xpUtils, "to_numpy", lambda X: X.numpy())
        setattr(xpUtils, "unique", lambda X: _unique_tf(X, tf))
        setattr(xpUtils, "copy", lambda X: tf.identity(X))
        setattr(xpUtils, "tile", tf.tile)
        xp = tf.experimental.numpy
    elif arrayModuleName == _CUPY:
        import cupy

        setattr(xpUtils, "coo_matrix", cupy.sparse.coo_matrix)
        setattr(xpUtils, "hstack", cupy.sparse.hstack)
        setattr(xpUtils, "norm", cupy.linalg.norm)
        setattr(xpUtils, "to_numpy", lambda X: X.get())
        setattr(xpUtils, "unique", lambda X: _unique(X, cupy))
        setattr(xpUtils, "copy", lambda X: X.copy())
        setattr(xpUtils, "tile", cupy.tile)
        xp = cupy
    else:
        import scipy

        setattr(xpUtils, "coo_matrix", scipy.sparse.coo_matrix)
        setattr(xpUtils, "hstack", scipy.sparse.hstack)
        setattr(xpUtils, "norm", numpy.linalg.norm)
        setattr(xpUtils, "to_numpy", lambda X: X)
        setattr(xpUtils, "unique", lambda X: _unique(X, cupy))
        setattr(xpUtils, "copy", lambda X: X.copy())
        setattr(xpUtils, "tile", numpy.tile)
    setattr(xpUtils, "asscarlar", lambda X: numpy.asscalar(xpUtils.to_numpy(X)))
    return xp, xpUtils


def get_array_module(X):
    arrayModuleName = _get_array_module_name(X)
    return get_array_module_with_utils(arrayModuleName)
