from sklearn.base import TransformerMixin
from Common import ReservoirSamplingEstimator, get_array_module


def _ann(X, samples, p):
    xp, _ = get_array_module(X)
    indices = xp.zeros(X.shape[0], dtype=int)
    l_dis_min = None
    for i in range(samples.shape[0]):
        i_sample = samples[i, :]
        l_dis = xp.sum((X - xp.expand_dims(i_sample, 0)) ** p, axis=1) ** (1 / p)
        if l_dis_min is None:
            l_dis_min = l_dis
        else:
            l_nearer = l_dis < l_dis_min
            # indices[l_nearer] = i
            indices = xp.where(l_nearer, i, indices)
            # l_dis_min[l_nearer] = l_dis[l_nearer]
            l_dis_min = xp.where(l_nearer, l_dis, l_dis_min)
    indices = xp.expand_dims(indices, axis=1)
    return indices, l_dis_min


class VoronoiPartitioning(ReservoirSamplingEstimator, TransformerMixin):
    def __init__(self, psi, metric="minkowski", p=2, **kwargs):
        super().__init__(psi)
        self.metric = metric
        self.p = p

    def partial_fit(self, X, y=None):
        super().partial_fit(X, y)
        return self

    def transform(self, X):
        if self.metric == "minkowski" and self.p > 0:
            indices, _ = _ann(X, self.samples_, self.p)
            return indices
        else:
            raise NotImplementedError()

    def score_samples(self, X):
        _, l_dis_min = _ann(X, self.samples_, self.p)
        return -l_dis_min
