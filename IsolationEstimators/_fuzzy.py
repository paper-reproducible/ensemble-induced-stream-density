import numpy as np
from sklearn.base import TransformerMixin
from Common import ReservoirSamplingEstimator, get_array_module


_gaussian = "gaussian"
_member_functions = [_gaussian]
_not_implemented = NotImplementedError("This member function is not supported.")


def _batch_gaussian(X, locs, scales):
    xp, xpUtils = get_array_module(X)
    _, dims = X.shape
    scales = xp.expand_dims(scales, axis=0)
    p = xpUtils.pdist(X, locs, root=False)  # [n, psi]
    p = xp.exp(p / (0 - 2 * (scales**2)))
    p = p / scales  # / xp.sqrt((np.pi * 2) ** dims)
    return p


class FuzzyPartitioning(ReservoirSamplingEstimator, TransformerMixin):
    def __init__(self, psi, member_function=_gaussian, random_scale=False, normalize=False, **kwargs):
        super().__init__(psi)
        if member_function not in _member_functions:
            raise _not_implemented
        self.member_function = member_function
        self.random_scale = random_scale
        self.normalize = normalize

    def partial_fit(self, X, y=None):
        super().partial_fit(X, y)
        return self

    def transform(self, X):
        xp, xpUtils = get_array_module(X)
        if self.member_function == _gaussian:
            locs = self.samples_  # [psi, dims]
            scales = xp.sort(xpUtils.pdist(locs, locs), axis=1)[:, 1]  # [psi]
            if self.random_scale:
                scales = scales * xp.random.rand()
            feature_map = _batch_gaussian(X, locs, scales)
            if self.normalize:
                feature_map = feature_map/xp.sum(feature_map)
            return feature_map
        else:
            raise _not_implemented
