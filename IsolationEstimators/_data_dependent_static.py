import numpy as np
from sklearn.base import DensityMixin, OutlierMixin
from joblib import delayed
from ._data_dependent import IsolationTransformer
from Common import get_array_module
from ._constants import IFOREST


class MassEstimator(IsolationTransformer, DensityMixin):
    def score(self, X, y=None):
        xp, _ = get_array_module(X)
        m_sim = self.transform(X, return_similarity=True)
        return xp.sum(m_sim, axis=1)


class IsolationBasedAnomalyDetector(MassEstimator, OutlierMixin):
    def __init__(
        self,
        psi,
        t,
        contamination="auto",
        mass_based=True,
        partitioning_type=IFOREST,
        n_jobs=16,
        verbose=0,
        parallel=None,
        **kwargs
    ):
        super().__init__(
            psi,
            t,
            partitioning_type,
            n_jobs,
            verbose,
            parallel,
            anomaly_detection=True,
            **kwargs
        )
        self.contamination = contamination
        if hasattr(self.base_transformer, "score_samples"):
            self.mass_based = mass_based
        else:
            self.mass_based = True
        self.offset_ = -0.5
        self.fitted_anomaly_score_ = None
        self.fitted_X_ = None

    def fit(self, X, y=None):
        super().fit(X, y)
        if self.contamination != "auto":
            _, xpUtils = get_array_module(X)
            anomaly_scores = self.score_samples(X)
            self.offset_ = xpUtils.percentile(
                anomaly_scores, 100.0 * self.contamination
            )
            self.fitted_anomaly_score_ = anomaly_scores
            self.fitted_X_ = X
        return self

    def predict(self, X, y=None):
        xp, xpUtils = get_array_module(X)
        if self.fitted_X_ is not None and xp.array_equal(self.fitted_X_, X):
            anomaly_scores = self.fitted_anomaly_score_
        else:
            anomaly_scores = self.score_samples(X)
        offset = self.offset_
        offsetted_scores = anomaly_scores - offset
        inliers = xp.ones_like(offsetted_scores, dtype=offsetted_scores.dtype)
        outliers = xp.where(offsetted_scores < 0)[0]
        if outliers.shape[0] > 0:
            inliers = xpUtils.tensor_scatter_nd_update(
                inliers, outliers, xp.array([-1], dtype=inliers.dtype)
            )
        return inliers

    def score_samples(self, X):
        if self.mass_based:
            return self.score(X)
        else:
            xp, _ = get_array_module(X)

            def loop_body(estimator):
                return estimator.score_samples(X)

            all_results = self.parallel()(
                delayed(loop_body)(i) for i in self.transformers_
            )
            return xp.average(xp.array(all_results), axis=0)
