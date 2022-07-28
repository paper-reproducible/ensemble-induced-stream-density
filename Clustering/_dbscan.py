import numpy as np
from Common import get_array_module
from scipy.sparse.csgraph import connected_components


def dbscan(m_dis, eps, minPts):
    xp, xpUtils = get_array_module(m_dis)
    n = m_dis.shape[0]
    m_adj = m_dis <= eps
    l_core = xp.sum(m_adj, axis=1) >= minPts

    n_core = xp.sum(l_core)
    if n_core <= 0:
        return xp.zeros([n]), 0

    m_adj_core = m_adj[l_core, :][:, l_core]

    n_components, labels_core = connected_components(xpUtils.to_numpy(m_adj_core))

    l_assign_core = xp.argmin(m_dis[:, l_core], axis=1)
    labels = xp.take(labels_core, l_assign_core)
    labels = xp.where(m_dis[xp.arange(n), l_assign_core] <= eps, labels, -1)

    return labels, n_components


from sklearn.base import BaseEstimator, ClusterMixin
from IsolationEstimators import IsolationTransformer

_ISOLATION = ["anne", "iforest", "fuzzi", "inne"]


class DBSCAN(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        eps,
        minPts,
        metric="minkowski",
        p=2,
        psi=None,
        t=1000,
        dtype=np.float32,
        parallel=None,
        **kwargs
    ):
        self.eps = eps
        self.minPts = minPts
        self.metric = metric
        self.p = p
        self.X_ = None
        self.dtype = dtype
        if self.metric in _ISOLATION:
            self.transformer_ = IsolationTransformer(
                psi,
                t,
                partitioning_type=metric,
                parallel=parallel,
                metric="minkowski",
                p=p,
                **kwargs,
            )
        elif metric != "minkowski":
            raise NotImplementedError()
        return

    def partial_fit(self, X, y=None):
        xp, xpUtils = get_array_module(X)
        if self.X_ is None:
            self.X_ = xpUtils.cast(X, dtype=self.dtype)
        else:
            self.X_ = xp.concatenate([self.X_, X], axis=0)
        if self.metric in _ISOLATION:
            self.transformer_.partial_fit(xpUtils.cast(X, dtype=self.dtype))
        return self

    def fit(self, X, y=None):
        return self.partial_fit(X, y)

    def predict(self, X, y=None):
        if X is not None:
            raise NotImplementedError()

        xp, xpUtils = get_array_module(self.X_)
        if self.metric == "minkowski":
            m_dis = xpUtils.norm(
                xp.expand_dims(self.X_, axis=0) - xp.expand_dims(self.X_, axis=1),
                ord=self.p,
                axis=2,
            )
        elif _ISOLATION.index(self.metric) >= 0:
            # encoded = self.transformer_.transform(self.X_)
            m_sim = self.transformer_.transform(self.X_, return_similarity=True)
            m_dis = xp.subtract(1, m_sim)
        else:
            raise NotImplementedError()

        # print(self.metric, m_dis * 1000)
        y, _ = dbscan(m_dis, self.eps, self.minPts)

        return y

    def fit_predict(self, X, y=None):
        return self.partial_fit(X, y).predict(X=None)


if __name__ == "__main__":
    # import tensorflow as tf

    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
    # tnp = tf.experimental.numpy
    # tnp.experimental_enable_numpy_behavior()

    # xp = tnp
    xp = np

    X = xp.expand_dims(xp.array([2, 3, 8, 9, 100], dtype=np.float64), axis=1)
    m = DBSCAN(eps=1.5, minPts=2)
    labels = m.fit_predict(X)
    print("l2:", labels)

    from joblib import Parallel

    np.set_printoptions(precision=2)

    # with Parallel(n_jobs=32, prefer="processes") as parallel:
    with Parallel(n_jobs=32, prefer="threads") as parallel:

        m = DBSCAN(eps=0.2, minPts=2, metric="inne", psi=2, t=200, parallel=parallel)
        labels = m.fit_predict(X)
        print("inne:", labels)

        m = DBSCAN(
            eps=0.99,
            minPts=2,
            metric="fuzzi",
            psi=2,
            t=200,
            parallel=parallel,
            random_scale=True,
        )
        labels = m.fit_predict(X)
        print("fuzzi:", labels)

        m = DBSCAN(eps=0.2, minPts=2, metric="anne", psi=2, t=200, parallel=parallel)
        labels = m.fit_predict(X)
        print("anne:", labels)

        from Common import ball_scale

        m = DBSCAN(eps=0.2, minPts=2, metric="iforest", psi=2, t=200, parallel=parallel)
        labels = m.fit_predict(ball_scale(X))
        print("iforest", labels)
