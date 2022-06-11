from cmath import asin
import numpy as np
from Common import get_array_module
from sklearn.base import BaseEstimator


def get_samples(X, psi):
    xp, xpUtils = get_array_module(X)
    X = xpUtils.unique(X)
    n = X.shape[0]

    if psi < n:
        new_indices = np.random.choice(n, size=(psi), replace=False).tolist()
        return xp.take(X, new_indices, axis=0)
    else:
        return xp.copy(X)


def update_samples(X, samples, start=0):
    xp, xpUtils = get_array_module(X)

    n = X.shape[0]
    psi = samples.shape[0]

    new_samples = None
    drop_samples = None

    replace_by = xp.ones(psi, dtype=int) * -1
    reservoir = xp.copy(samples)

    r = xp.random.rand(n)
    r = xp.floor((xp.arange(n, dtype=r.dtype) + start) * r)

    for j in range(psi):
        potential_i = xp.where(r == j)[0]
        if potential_i.shape[0] == 0:
            continue
        i = xp.max(potential_i)
        item = X[i : i + 1, :]
        if xp.any(xp.all(reservoir == item, axis=1)):
            continue  # no duplicates
        replace_by = xp.where(xp.arange(replace_by.shape[0]) == j, i, replace_by)
        reservoir = xp.where(
            xp.expand_dims(xp.arange(reservoir.shape[0]), axis=1) == j,
            item[0, :],
            reservoir,
        )

    drop_indices = replace_by >= 0
    changed_count = xp.sum(drop_indices)
    if changed_count > 0:
        drop_samples = xp.take(samples, xp.where(drop_indices)[0], axis=0)
        new_samples = xp.take(X, replace_by[drop_indices], axis=0)

    return int(xpUtils.asscalar(changed_count)), new_samples, drop_samples, reservoir


class ReservoirSamplingEstimator(BaseEstimator):
    def __init__(self, psi):
        self.psi = psi
        self.fitted = 0

    def fit(self, X, y=None):
        self.samples_ = get_samples(X, self.psi)
        self.fitted = X.shape[0]
        self.changed_ = True
        return self

    def partial_fit(self, X, y=None):
        if self.fitted == 0:
            return self.fit(X, y), self.psi, self.samples_, None
        changed, _, _, reservoir = self.update_samples(X)
        self.samples_ = reservoir
        self.changed_ = changed > 0
        self.fitted = self.fitted + X.shape[0]
        return self

    def update_samples(self, X):
        return update_samples(X, self.samples_, start=self.fitted)
