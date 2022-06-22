from sklearn.base import TransformerMixin, DensityMixin
from joblib import delayed
from ._bagging import BaseAdaptiveBaggingEstimator
from ._voronoi import VoronoiPartitioning
from ._isolation_tree import IsolationTree, IncrementalMassEstimationTree
from Common import get_array_module
from ._constants import ANNE, IFOREST


class IsolationTransformer(BaseAdaptiveBaggingEstimator, TransformerMixin):
    def __init__(
        self,
        psi,
        t,
        partitioning_type=IFOREST,
        n_jobs=16,
        verbose=0,
        parallel=None,
        metric="minkowski",
        p=2,
    ):
        if partitioning_type == ANNE:
            base_transformer = VoronoiPartitioning(psi, metric, p)
        elif partitioning_type == IFOREST:
            base_transformer = IsolationTree(psi)
        else:
            raise NotImplementedError()
        super().__init__(base_transformer, t, n_jobs, verbose, parallel)
        self.psi = psi
        self.partitioning_type = partitioning_type

    def transform(self, X, return_similarity=False):
        xp, xpUtils = get_array_module(X)
        n = X.shape[0]

        if return_similarity:

            def loop_body(estimator):
                indices = estimator.transform(X)
                return xp.equal(indices, xp.transpose(indices))

            all_results = self.parallel()(
                delayed(loop_body)(i) for i in self.transformers_
            )
            return xp.average(xp.array(all_results), axis=0)
        else:

            def loop_body(estimator):
                indices = estimator.transform(X)
                encoded = xpUtils.coo_matrix(
                    (xp.ones(n, dtype=float), (xp.arange(n), indices[:, 0])),
                    shape=(n, self.psi),
                )
                return encoded

            all_results = self.parallel()(
                delayed(loop_body)(i) for i in self.transformers_
            )
            return xpUtils.hstack(all_results)


class MassEstimator(BaseAdaptiveBaggingEstimator, DensityMixin):
    def __init__(
        self,
        psi,
        t,
        partitioning_type=IFOREST,
        n_jobs=16,
        verbose=0,
        parallel=None,
        rotation=True,
    ):
        if partitioning_type == IFOREST:
            base_transformer = IncrementalMassEstimationTree(psi, rotation)
        else:
            raise NotImplementedError()
        super().__init__(base_transformer, t, n_jobs, verbose, parallel)
        self.psi = psi

    def score(self, X, return_demass=False):
        xp, _ = get_array_module(X)

        def loop_body(estimator):
            return estimator.score(X, return_demass)

        all_results = self.parallel()(delayed(loop_body)(i) for i in self.transformers_)
        return xp.average(xp.array(all_results), axis=0)


class DEMassEstimator(MassEstimator):
    def __init__(
        self,
        psi,
        t,
        partitioning_type=IFOREST,
        n_jobs=16,
        verbose=0,
        parallel=None,
        rotation=True,
    ):
        super().__init__(psi, t, partitioning_type, n_jobs, verbose, parallel, rotation)

    def score(self, X):
        return super().score(X, return_demass=True)
