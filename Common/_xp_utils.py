# pyright: reportMissingImports=false

import importlib
import importlib.util
import sys
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


def _pdist(X, Y, p=2, root=True, xp=numpy, xpUtils=None):
    if xpUtils is None or not hasattr(xpUtils, "norm"):
        result = xp.sum(
            (xp.expand_dims(X, axis=1) - xp.expand_dims(Y, axis=0)) ** p,
            axis=2,
        )
    else:
        result = xpUtils.norm(
            xp.expand_dims(X, axis=1) - xp.expand_dims(Y, axis=0),
            ord=p,
            axis=2,
        )
    if root:
        result = result ** (1 / p)
    return result


def get_array_module_with_utils(arrayModuleName):
    utilsModuleName = _XPUTILS + "." + arrayModuleName
    if utilsModuleName in sys.modules:
        return sys.modules[utilsModuleName]
    spec = importlib.machinery.ModuleSpec(utilsModuleName, None)
    xp = numpy
    xpUtils = importlib.util.module_from_spec(spec)
    if arrayModuleName == _TNP:
        from ._xp_utils_tf import setup_tf

        xp = setup_tf(xpUtils)
    elif arrayModuleName == _CUPY:
        from ._xp_utils_xp import setup_cupy

        xp = setup_cupy(xpUtils)
    else:
        from ._xp_utils_xp import setup_numpy

        xp = setup_numpy(xpUtils)
    setattr(xpUtils, "asscalar", lambda X: numpy.asscalar(xpUtils.to_numpy(X)))
    setattr(xpUtils, "pdist", lambda X, Y, p=2, root=True: _pdist(X, Y, p, root, xp))
    return xp, xpUtils


def get_array_module(X):
    arrayModuleName = _get_array_module_name(X)
    return get_array_module_with_utils(arrayModuleName)
