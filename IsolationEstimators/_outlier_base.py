from abc import abstractmethod
from sklearn.base import BaseEstimator, OutlierMixin
from Common import get_array_module


class BaseAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, contamination="auto"):
        self.contamination = contamination
        self.offset_ = -0.5
        self.fitted_anomaly_score_ = None
        self.fitted_X_ = None

    def fit(self, X, y=None):
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

    @abstractmethod
    def score_samples(self, X):
        pass
