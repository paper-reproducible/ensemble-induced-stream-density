import numpy as np
from sklearn.base import DensityMixin
from joblib import delayed
from ._bagging import BaseAdaptiveBaggingEstimator
from ._isolation_tree import IsolationTree
from ._voronoi import VoronoiPartitioning
from ._voronoi_soft import SoftVoronoiPartitioning
from ._fuzzy import FuzziPartitioning
from ._inn import INNPartitioning
from Common import get_array_module, ball_samples
from ._constants import ANNE, IFOREST, FUZZI, INNE, SOFT_ANNE


def _single_fit(transformer, X):
    dims = X.shape[1]
    xp, xpUtils = get_array_module(X)
    samples = ball_samples(transformer.psi, dims, xp=xp, linalg=xpUtils)
    transformer.fit(samples)
    indices = transformer.transform(X)
    transformer.region_mass_ = xp.sum(
        xp.equal(indices, xp.expand_dims(xp.arange(transformer.psi), axis=0)),
        axis=0,
    )
    return transformer


def _single_partial_fit(transformer, X):
    xp, _ = get_array_module(X)
    indices = transformer.transform(X)
    transformer.region_mass_ = transformer.region_mass_ + xp.sum(
        xp.equal(indices, xp.expand_dims(xp.arange(transformer.psi), axis=0)),
        axis=0,
    )
    return transformer


def _single_score(partitioning_type, transformer, X, return_demass):
    xp, xpUtils = get_array_module(X)
    region_mass = transformer.region_mass_
    indices = transformer.transform(X)
    if return_demass:
        if partitioning_type == IFOREST:
            region_volumes = transformer.node_volumes_[transformer.node_is_leaf_]
        else:
            # Never gonna happen
            region_volumes = transformer.region_volumes_
        region_demass = (
            xpUtils.cast(region_mass, dtype=np.dtype(float)) / region_volumes
        )
        return xp.take(region_demass, xp.squeeze(indices, axis=1))
    else:
        return xp.take(region_mass, xp.squeeze(indices, axis=1))


class DataIndependentEstimator(BaseAdaptiveBaggingEstimator, DensityMixin):
    def __init__(
        self,
        psi,
        t,
        partitioning_type=IFOREST,
        n_jobs=16,
        verbose=0,
        parallel=None,
        **kwargs
    ):
        if partitioning_type == IFOREST:
            transformer_factory = lambda: IsolationTree(psi, **kwargs)
        elif partitioning_type == ANNE:
            transformer_factory = lambda: VoronoiPartitioning(psi, **kwargs)
        elif partitioning_type == FUZZI:
            transformer_factory = lambda: FuzziPartitioning(psi, **kwargs)
        elif partitioning_type == INNE:
            transformer_factory = lambda: INNPartitioning(psi, **kwargs)
        elif partitioning_type == SOFT_ANNE:
            transformer_factory = lambda: SoftVoronoiPartitioning(psi, **kwargs)
        else:
            raise NotImplementedError()
        super().__init__(transformer_factory, t, n_jobs, verbose, parallel)
        self.psi = psi
        self.partitioning_type = partitioning_type

    def fit(self, X, y=None):
        xp, xpUtils = get_array_module(X)
        if xp.any(xpUtils.norm(X, axis=1) > 1):
            raise NotImplementedError("The data need to be ball_scale-ed")

        self.transformers_ = self.parallel()(
            delayed(_single_fit)(i, X) for i in self.transformers_
        )

        self.fitted = X.shape[0]
        return self

    def partial_fit(self, X, y=None):
        xp, xpUtils = get_array_module(X)
        if xp.any(xpUtils.norm(X, axis=1) > 1):
            raise NotImplementedError("The data need to be ball_scale-ed")

        if self.fitted == 0:
            return self.fit(X, y)

        self.transformers_ = self.parallel()(
            delayed(_single_partial_fit)(i, X) for i in self.transformers_
        )

        self.fitted = self.fitted + X.shape[0]
        return self

    def score(self, X, return_demass=False):
        xp, xpUtils = get_array_module(X)
        if xp.any(xpUtils.norm(X, axis=1) > 1):
            raise NotImplementedError("The data need to be ball_scale-ed")

        if return_demass and self.partitioning_type != IFOREST:
            return NotImplementedError()

        all_results = self.parallel()(
            delayed(_single_score)(self.partitioning_type, i, X, return_demass)
            for i in self.transformers_
        )
        return xp.average(xp.array(all_results), axis=0)


class DataIndependentDensityEstimator(DataIndependentEstimator):
    def __init__(
        self, psi, t, partitioning_type=IFOREST, n_jobs=16, verbose=0, parallel=None
    ):
        if partitioning_type != IFOREST:
            raise NotImplementedError()
        super().__init__(psi, t, partitioning_type, n_jobs, verbose, parallel)

    def score(self, X):
        return super().score(X, return_demass=True)
