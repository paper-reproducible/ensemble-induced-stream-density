from sklearn.base import TransformerMixin
from Common import ReservoirSamplingEstimator, get_array_module


class VoronoiPartitioning(ReservoirSamplingEstimator, TransformerMixin):
    def __init__(self, psi, metric="minkowski", p=2):
        super().__init__(psi)
        self.metric = metric
        self.p = p

    def partial_fit(self, X, y=None):
        super().partial_fit(X, y)
        return self

    def transform(self, X):
        xp, _ = get_array_module(X)
        if self.metric == "minkowski" and self.p > 0:
            indices = xp.zeros(X.shape[0], dtype=xp.int)
            l_dis_min = None
            for i in range(self.psi):
                i_sample = self.samples_[i, :]
                l_dis = xp.sum((X - xp.expand_dims(i_sample, 0)) ** self.p, axis=1) ** (
                    1 / self.p
                )
                if l_dis_min is None:
                    l_dis_min = l_dis
                else:
                    l_nearer = l_dis < l_dis_min
                    indices[l_nearer] = i
                    l_dis_min[l_nearer] = l_dis[l_nearer]
            indices = xp.expand_dims(indices, axis=0)
            return indices
        else:
            raise NotImplementedError()
