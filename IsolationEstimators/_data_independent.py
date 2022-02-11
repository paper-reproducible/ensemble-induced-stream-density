from sklearn.base import DensityMixin
from joblib import delayed
from ._bagging import BaseAdaptiveBaggingEstimator
from ._isolation_tree import IsolationTree
from ._voronoi import VoronoiPartitioning
from Common import get_array_module, ball_samples
from ._constants import ANNE, IFOREST


def _single_fit(e, transformer, X):
    dims = X.shape[1]
    xp, _ = get_array_module(X)
    samples = ball_samples(e.psi, dims, xp=xp)
    transformer.fit(samples)
    indices = transformer.transform(X)
    transformer.region_mass_ = xp.sum(
        xp.equal(indices, xp.expand_dims(xp.arange(transformer.psi), axis=1)), axis=1,
    )
    return


def _single_partial_fit(e, transformer, X):
    xp, _ = get_array_module(X)
    indices = transformer.transform(X)
    transformer.region_mass_ = transformer.region_mass_ + xp.sum(
        xp.equal(indices, xp.expand_dims(xp.arange(transformer.psi), axis=1)), axis=1,
    )
    return


def _single_score(e, transformer, X, return_demass):
    xp, _ = get_array_module(X)
    region_mass = transformer.region_mass_
    indices = transformer.transform(X)
    if return_demass:
        if e.partitioning_type == IFOREST:
            region_volumes = transformer.node_volumes_[transformer.node_is_leaf_]
        else:
            # Never gonna happen
            region_volumes = transformer.region_volumes_
        region_demass = region_mass / region_volumes
        return xp.squeeze(region_demass[indices], axis=0)
    else:
        return xp.squeeze(region_mass[indices], axis=0)


class DataIndependentEstimator(BaseAdaptiveBaggingEstimator, DensityMixin):
    def __init__(
        self, psi, t, partitioning_type=IFOREST, n_jobs=16, verbose=0, parallel=None
    ):
        if partitioning_type == IFOREST:
            base_transformer = IsolationTree(psi)
        elif partitioning_type == ANNE:
            base_transformer = VoronoiPartitioning(psi)
        else:
            raise NotImplementedError()
        super().__init__(base_transformer, t, n_jobs, verbose, parallel)
        self.psi = psi
        self.partitioning_type = partitioning_type

    def fit(self, X, y=None, ball_scaled=True):
        if not ball_scaled:
            raise NotImplementedError()

        self.parallel()(delayed(_single_fit)(self, i, X) for i in self.transformers_)

        self.fitted = X.shape[0]
        return self

    def partial_fit(self, X, y=None):
        if self.fitted == 0:
            return self.fit(X, y)

        self.parallel()(
            delayed(_single_partial_fit)(self, i, X) for i in self.transformers_
        )

        self.fitted = self.fitted + X.shape[0]
        return self

    def score(self, X, return_demass=False):
        xp, _ = get_array_module(X)

        if return_demass and self.partitioning_type != IFOREST:
            return NotImplementedError()

        all_results = self.parallel()(
            delayed(_single_score)(self, i, X, return_demass)
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
