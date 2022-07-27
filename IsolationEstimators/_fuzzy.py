import numpy as np
from sklearn.base import TransformerMixin
from Common import ReservoirSamplingEstimator, get_array_module


_gaussian = "gaussian"
_kernels = [_gaussian]


def _batch_gaussian(X, locs, scales):
    xp, xpUtils = get_array_module(X)
    _, dims = X.shape
    scales = xp.expand_dims(scales, axis=0)
    p = xpUtils.pdist(X, locs, root=False)  # [n, psi]
    p = xp.exp(p / (0 - 2 * (scales**2)))
    p = p / scales / xp.sqrt((np.pi * 2) ** dims)
    return p


class FuzziPartitioning(ReservoirSamplingEstimator, TransformerMixin):
    def __init__(self, psi, kernel=_gaussian, random_scale=False, **kwargs):
        super().__init__(psi)
        if kernel not in _kernels:
            raise NotImplementedError("This kernel is not supported.")
        self.kernel = kernel
        self.random_scale = random_scale

    def partial_fit(self, X, y=None):
        super().partial_fit(X, y)
        return self

    def transform(self, X):
        xp, xpUtils = get_array_module(X)
        if self.kernel == _gaussian:
            locs = self.samples_  # [psi, dims]
            scales = xp.sort(xpUtils.pdist(locs, locs), axis=1)[:, 1]  # [psi]
            if self.random_scale:
                scales = scales * xp.random.rand()
            return _batch_gaussian(X, locs, scales)
        else:
            raise NotImplementedError("This kernel is not supported.")
