import numpy as np
from sklearn.base import TransformerMixin, DensityMixin
from joblib import delayed
from ._bagging import BaseAdaptiveBaggingEstimator
from ._voronoi import VoronoiPartitioning
from ._isolation_tree import IsolationTree, IncrementalMassEstimationTree
from ._fuzzy import FuzziPartitioning
from ._inn import INNPartitioning
from Common import get_array_module
from ._constants import ANNE, IFOREST, FUZZI, INNE


class IsolationTransformer(BaseAdaptiveBaggingEstimator, TransformerMixin):
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
        if partitioning_type == ANNE:
            base_transformer = VoronoiPartitioning(psi, **kwargs)
        elif partitioning_type == IFOREST:
            base_transformer = IsolationTree(psi, **kwargs)
        elif partitioning_type == FUZZI:
            base_transformer = FuzziPartitioning(psi, **kwargs)
        elif partitioning_type == INNE:
            base_transformer = INNPartitioning(psi, **kwargs)
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
                if indices.shape[1] == 1:
                    return xp.equal(indices, xp.transpose(indices))
                elif indices.shape[1] == self.psi:
                    return xp.matmul(indices, xp.transpose(indices))

            all_results = self.parallel()(
                delayed(loop_body)(i) for i in self.transformers_
            )
            return xp.average(xp.array(all_results), axis=0)
        else:

            def loop_body(estimator):
                indices = estimator.transform(X)
                if indices.shape[1] == 1:
                    encoded = xpUtils.coo_matrix(
                        (
                            xp.ones(n, dtype=float),
                            (xp.arange(n), indices[:, 0]),
                        ),
                        shape=(n, self.psi),
                    )
                    return encoded
                elif indices.shape[1] == self.psi:
                    return indices

            all_results = self.parallel()(
                delayed(loop_body)(i) for i in self.transformers_
            )
            return xpUtils.hstack(all_results)


class IncrementalMassEstimator(BaseAdaptiveBaggingEstimator, DensityMixin):
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
        else:  # aNNE has no incremental implementation
            raise NotImplementedError()
        super().__init__(base_transformer, t, n_jobs, verbose, parallel)
        self.psi = psi

    def score(self, X, return_demass=False):
        xp, _ = get_array_module(X)

        def loop_body(estimator):
            return estimator.score(X, return_demass)

        all_results = self.parallel()(delayed(loop_body)(i) for i in self.transformers_)
        return xp.average(xp.array(all_results), axis=0)


class DEMassEstimator(IncrementalMassEstimator):
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


class MassEstimator(IsolationTransformer, DensityMixin):
    def score(self, X, y=None):
        xp, _ = get_array_module(X)
        m_sim = self.transform(X, return_similarity=True)
        return xp.sum(m_sim, axis=1)
