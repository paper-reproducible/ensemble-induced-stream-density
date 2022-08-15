import numpy as np
from Common import get_array_module
from scipy.sparse.csgraph import connected_components


def _adj(m_sim, eps, using_similarity=False):
    if using_similarity:
        m_adj = m_sim >= eps
    else:
        m_adj = m_sim <= eps
    return m_adj


def _dens(m_adj, xp=np):
    return xp.sum(m_adj, axis=1)


def epsilon_neighbourhood_density(m_sim, eps, using_similarity=False, xp=np):
    m_adj = _adj(m_sim, eps, using_similarity)
    return _dens(m_adj, xp)


def _dbscan(m_sim, eps, threshold, l_dens=None, using_similarity=False):
    xp, xpUtils = get_array_module(m_sim)
    n = m_sim.shape[0]

    m_adj = _adj(m_sim, eps, using_similarity)
    if l_dens is None:
        l_dens = _dens(m_adj, xp)

    # print(m_sim)
    # print(l_dens)
    l_core = l_dens >= threshold

    # All points are noise if there is no core points.
    if xp.sum(l_core) <= 0:
        return xp.zeros([n]) - 1, 0

    # Label the core points by extracting the connected components of the \epsilon-neighbourhood graph.
    m_adj_core = m_adj[l_core, :][:, l_core]
    n_components, labels_core = connected_components(xpUtils.to_numpy(m_adj_core))
    labels_core = xp.array(labels_core)

    # Assign labels to all points using the label of their nearest core points.
    if using_similarity:
        l_assign_core = xp.argmax(m_sim[:, l_core], axis=1)
    else:
        l_assign_core = xp.argmin(m_sim[:, l_core], axis=1)
    labels = xp.take(labels_core, l_assign_core)

    # Label points as anomaly if their are outside of the \epsilon-neighbourhood of their nearest core points.
    if using_similarity:
        labels = xp.where(m_sim[xp.arange(n), l_assign_core] >= eps, labels + 1, -1)
    else:
        labels = xp.where(m_sim[xp.arange(n), l_assign_core] <= eps, labels + 1, -1)

    return labels, n_components


def dbscan(m_dis, eps, minPts):
    labels, n_components = _dbscan(
        m_dis, eps, minPts, l_dens=None, using_similarity=False
    )
    return labels, n_components


from sklearn.base import BaseEstimator, ClusterMixin
from IsolationEstimators import EstimatorType, IsolationModel, init_estimator


class DBSCAN(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        eps,
        core_threshold,
        psi,
        isolation_model_name=None,
        use_alpha_neighbourhood_mass=True,
        use_anomaly_score=False,
        dtype=np.float32,
        **kwargs
    ):
        self.eps = eps
        self.core_threshold = core_threshold
        self.isolation_model_name = isolation_model_name
        self.use_alpha_neighbourhood_mass = use_alpha_neighbourhood_mass
        self.use_anomaly_score = use_anomaly_score

        if isolation_model_name is None:
            self.isolation_model = None
        elif use_alpha_neighbourhood_mass:
            self.isolation_model = init_estimator(
                EstimatorType.KERNEL,
                IsolationModel(isolation_model_name),
                psi,
                **kwargs
            )
        elif use_anomaly_score:
            self.isolation_model = init_estimator(
                EstimatorType.ANOMALY,
                IsolationModel(isolation_model_name),
                psi,
                mass_based=False,
                **kwargs
            )
        else:
            self.isolation_model = init_estimator(
                EstimatorType.MASS, IsolationModel(isolation_model_name), psi, **kwargs
            )

        self.dtype = dtype
        self.X_ = None
        return

    def fit(self, X, y=None):
        xp, xpUtils = get_array_module(X)
        self.X_ = xpUtils.cast(X, dtype=self.dtype)

        # The similarity matrix and density are calculated for tunning purpose.
        if self.isolation_model is not None:
            self.isolation_model.fit(self.X_)

        using_similarity = (
            self.isolation_model is not None and self.use_alpha_neighbourhood_mass
        )
        if using_similarity:
            m_sim = self.isolation_model.transform(self.X_, return_similarity=True)
        else:
            m_sim = xpUtils.norm(
                xp.expand_dims(self.X_, axis=0) - xp.expand_dims(self.X_, axis=1),
                ord=2,
                axis=2,
            )

        if self.use_alpha_neighbourhood_mass:
            l_dens = None
        elif self.isolation_model is not None:
            if self.use_anomaly_score:
                l_dens = self.isolation_model.score_samples(self.X_)
            else:
                l_dens = self.isolation_model.score(self.X_)
        else:
            l_dens = None

        self.using_similarity = using_similarity
        self.m_sim = m_sim
        self.l_dens = l_dens
        return self

    def predict(self, X, y=None):
        xp, _ = get_array_module(self.X_)
        if X is not None and xp.any(self.X_ != X):
            raise NotImplementedError()

        using_similarity = (
            self.isolation_model is not None and self.use_alpha_neighbourhood_mass
        )
        y, _ = _dbscan(
            self.m_sim, self.eps, self.core_threshold, self.l_dens, using_similarity
        )
        return y

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X=None)


if __name__ == "__main__":
    from Common import ball_scale

    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    xp = np

    # import os
    # import tensorflow as tf

    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
    # tnp = tf.experimental.numpy
    # tnp.experimental_enable_numpy_behavior()
    # xp = tnp

    X = xp.array(
        [[-5.1, -5.1], [-4.6, -4.6], [-1.1, -1.1], [-0.6, -0.8], [6.6, 6.6]],
        dtype=np.float64,
    )
    y_true = xp.array([1, 1, 2, 2, -1])
    X_ball = ball_scale(X)

    from joblib import Parallel

    np.set_printoptions(precision=2)

    isolation_models = [None] + [e.value for e in IsolationModel]

    params = {
        "l2_similarity": (1, 2, None),
        "anne_similarity": (0.8, 2, 2),
        "soft_anne_similarity": (0.8, 2, 2),
        "inne_similarity": (0.2, 2, 2),
        "isotropic_similarity": (0.1, 2, 2),
        "iforest_similarity": (0.8, 2, 2),
        "anne_anomaly": (0.8, -3, 2),
        "soft_anne_anomaly": (0.8, -3, 2),
        "inne_anomaly": (0.8, -0.4, 2),
        "iforest_anomaly": (1, 1.59, 3),
        "anne_mass": (0.8, 2.5, 2),
        "soft_anne_mass": (0.8, 2.5, 2),
        "inne_mass": (1, 1, 2),
        "isotropic_mass": (1, 2, 3),
        "iforest_mass": (1, 2.5, 2),
    }
    # isotropic_anomaly does not exist.

    with Parallel(n_jobs=32, prefer="threads") as parallel:
        for name in params:

            splits = name.split("_")
            model_name = "_".join(splits[:-1])
            suffix = splits[len(splits) - 1]

            eps, minPts, psi = params[name]

            rotation = False  # Tired to tune rotated iforest

            X_ = (
                X_ball
                if rotation
                and model_name is not None
                and model_name.startswith("iforest")
                else X
            )

            if suffix == "similarity":
                m = DBSCAN(
                    eps,
                    minPts,
                    psi,
                    isolation_model_name=None if model_name == "l2" else model_name,
                    use_alpha_neighbourhood_mass=True,
                    rotation=rotation,
                    parallel=parallel,
                )
                labels = m.fit_predict(X_)
                print(
                    model_name
                    + (
                        "_epsilon_neighbourhood"
                        if model_name == "l2"
                        else "_alpha_neighbourhood"
                    ),
                    ">>",
                    labels,
                )
                if not xp.all(labels == y_true):
                    print("Wrong results!")

            if suffix == "anomaly":
                m = DBSCAN(
                    eps,
                    minPts,
                    psi,
                    isolation_model_name=model_name,
                    use_alpha_neighbourhood_mass=False,
                    use_anomaly_score=True,
                    rotation=rotation,
                    parallel=parallel,
                )
                labels = m.fit_predict(X_)
                print(model_name + "_anomaly", ">>", labels)
                if not xp.all(labels == y_true):
                    print("Wrong results!")

            if suffix == "mass":
                m = DBSCAN(
                    eps,
                    minPts,
                    psi,
                    isolation_model_name=model_name,
                    use_alpha_neighbourhood_mass=False,
                    use_anomaly_score=False,
                    rotation=rotation,
                    parallel=parallel,
                )
                labels = m.fit_predict(X_)
                print(model_name + "_mass", ">>", labels)
                if not xp.all(labels == y_true):
                    print("Wrong results!")
