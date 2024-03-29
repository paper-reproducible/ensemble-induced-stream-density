import numpy as np
from sklearn.base import TransformerMixin, DensityMixin
from joblib import delayed
from ._bagging import BaseAdaptiveBaggingEstimator
from ._isolation_tree import AdaptiveMassEstimationTree
from Common import get_array_module
from ._naming import IsolationModel, get_partitioning_initializer


class DataDependentEstimator(BaseAdaptiveBaggingEstimator):
    def fit(self, X, y=None):
        def single_fit(bagger, X):
            e = bagger.partitioning_initializer()
            e.fit(X)
            return e

        self.transformers_ = self.parallel()(
            delayed(single_fit)(self, X) for _ in range(self.t)
        )
        self.fitted = X.shape[0]
        return self


class IsolationTransformer(DataDependentEstimator, TransformerMixin):
    def __init__(
        self,
        psi,
        t,
        isolation_model=IsolationModel.IFOREST.value,
        n_jobs=16,
        verbose=0,
        parallel=None,
        **kwargs
    ):
        partitioning_initializer = get_partitioning_initializer(
            isolation_model, psi, **kwargs
        )
        super().__init__(partitioning_initializer, t, n_jobs, verbose, parallel)
        self.psi = psi
        self.isolation_model = isolation_model

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


class IncrementalMassEstimator(DataDependentEstimator, DensityMixin):
    def __init__(self, psi, t, n_jobs=16, verbose=0, parallel=None, **kwargs):
        partitioning_initializer = lambda: AdaptiveMassEstimationTree(psi, **kwargs)

        super().__init__(
            partitioning_initializer, t, n_jobs, verbose, parallel, **kwargs
        )
        self.psi = psi

    def score(self, X, return_demass=False):
        xp, _ = get_array_module(X)

        def loop_body(estimator):
            result = estimator.score(X, return_demass)
            # if hasattr(estimator, "backup"):
            #     boundaries = estimator.combine_boundaries()
            #     print(xp.all(boundaries == estimator.backup))
            return result

        all_results = self.parallel()(delayed(loop_body)(i) for i in self.transformers_)
        result = xp.average(xp.array(all_results), axis=0)
        return result


class DEMassEstimator(IncrementalMassEstimator):
    def __init__(self, psi, t, n_jobs=16, verbose=0, parallel=None, **kwargs):
        super().__init__(psi, t, n_jobs, verbose, parallel, **kwargs)

    def score(self, X):
        return super().score(X, return_demass=True)


class MassEstimator(IsolationTransformer, DensityMixin):
    def fit(self, X, y=None):
        super().fit(X, y)
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        def loop_body(transformer):
            xp, _ = get_array_module(X)
            psi = transformer.psi
            indices = transformer.transform(X)
            if indices.shape[1] == 1:
                encoded = xp.expand_dims(xp.arange(psi), axis=0) == indices
            elif indices.shape[1] == transformer.psi:
                encoded = indices

            if hasattr(transformer, "region_mass_"):
                transformer.region_mass_ += xp.sum(encoded, axis=0)
            else:
                transformer.region_mass_ = xp.sum(encoded, axis=0)
            return transformer

        self.parallel()(delayed(loop_body)(i) for i in self.transformers_)
        return self

    def score(self, X, y=None):
        xp, _ = get_array_module(X)

        # m_sim = self.transform(X, return_similarity=True)
        # return xp.sum(m_sim, axis=1)
        # The above method causes OOM.

        def loop_body(transformer):
            xp, _ = get_array_module(X)
            psi = transformer.psi
            indices = transformer.transform(X)
            if indices.shape[1] == 1:
                encoded = xp.expand_dims(xp.arange(psi), axis=0) == indices
            elif indices.shape[1] == transformer.psi:
                encoded = indices
            region_mass = transformer.region_mass_
            X_mass = xp.sum(encoded * xp.expand_dims(region_mass, 0), axis=1)
            return X_mass

        all_results = self.parallel()(delayed(loop_body)(i) for i in self.transformers_)
        return xp.average(xp.array(all_results), axis=0)
