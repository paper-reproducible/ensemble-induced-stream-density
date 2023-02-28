import numpy as np
from sklearn.base import TransformerMixin
from Common import ReservoirSamplingEstimator, get_array_module


def _inn(X, c, r):
    xp, xpUtils = get_array_module(X)
    x_dists = xpUtils.pdist(X, c, root=False)
    cover_radius = xp.where(x_dists <= r, r, np.nan)
    x_covered = xp.where(xp.any(cover_radius < xp.inf, axis=1))[0]
    cnn_x = xp.argmin(cover_radius[x_covered], axis=1)
    cover_radius = xp.where(x_dists <= r, r, xp.inf)
    return x_covered, cnn_x


class INNPartitioning(ReservoirSamplingEstimator, TransformerMixin):
    def __init__(self, psi, anomaly_detection=False, **kwargs):
        super().__init__(psi)
        self.anomaly_detection = anomaly_detection
        self.region_scale_ = None
        if self.anomaly_detection:
            self.ratio_ = None  # xp.ones([psi])

    def _fit(self, X):
        xp, xpUtils = get_array_module(X)
        centroids = self.samples_  # [psi, dims]

        center_dist = xpUtils.pdist(centroids, centroids)
        center_dist = center_dist + xp.diag(xp.ones(self.psi, dtype=X.dtype) * xp.inf)
        centroids_radius = xp.amin(center_dist, axis=1)
        self.region_scale_ = centroids_radius

        # for anomaly detection:
        if self.anomaly_detection:
            eps = xp.finfo(X.dtype).eps
            cnn_index = xp.argmin(center_dist, axis=1)
            cnn_radius = centroids_radius[cnn_index]
            self.ratio_ = 1 - (cnn_radius + eps) / (centroids_radius + eps)
        return self

    def fit(self, X, y=None):
        super().fit(X, y)
        return self._fit(X)

    def partial_fit(self, X, y=None):
        super().partial_fit(X, y)
        return self._fit(X)

    def transform(self, X):
        xp, xpUtils = get_array_module(X)
        centroids = self.samples_
        centroids_radius = self.region_scale_

        x_covered, cnn_x = _inn(X, centroids, centroids_radius)
        # encoded = xpUtils.coo_matrix(
        #     (
        #         xp.ones(x_covered.shape[0], dtype=X.dtype),
        #         (x_covered, cnn_x),
        #     ),
        #     shape=(X.shape[0], self.psi),
        # )
        indices = xpUtils.tensor_scatter_nd_update(
            xp.full([X.shape[0]], np.nan, dtype=np.float32),
            x_covered,
            xpUtils.cast(cnn_x, dtype=np.float32),
        )  # int array cannot have nan
        return xp.expand_dims(indices, axis=1)

    def score_samples(self, X):
        xp, xpUtils = get_array_module(X)
        centroids = self.samples_
        centroids_radius = self.region_scale_
        ratio = self.ratio_

        x_covered, cnn_x = _inn(X, centroids, centroids_radius)
        isolation_scores = xpUtils.tensor_scatter_nd_update(
            xp.ones([X.shape[0]], dtype=ratio.dtype), x_covered, ratio[cnn_x]
        )
        return -isolation_scores


if __name__ == "__main__":

    # import os

    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
    # import tensorflow as tf

    # tnp = tf.experimental.numpy
    # tnp.experimental_enable_numpy_behavior()
    # xp = tnp
    xp = np

    X = xp.array([[2.1], [3.1], [8.1], [9.1], [100.1]])

    from joblib import Parallel
    from IsolationEstimators import IsolationBasedAnomalyDetector

    np.set_printoptions(precision=2)
    with Parallel(n_jobs=32, prefer="threads") as parallel:
        e = IsolationBasedAnomalyDetector(
            2,
            1000,
            contamination=0.2,
            mass_based=False,
            isolation_model="inne",
            parallel=parallel,
        )
        result = e.fit_predict(X)
        print("By ratio: ", result.numpy() if hasattr(result, "numpy") else result)

        e = IsolationBasedAnomalyDetector(
            2,
            1000,
            contamination=0.2,
            mass_based=True,
            isolation_model="inne",
            parallel=parallel,
        )
        result = e.fit_predict(X)
        print("By mass: ", result.numpy() if hasattr(result, "numpy") else result)
