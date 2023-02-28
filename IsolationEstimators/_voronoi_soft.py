from sklearn.base import TransformerMixin
from Common import ReservoirSamplingEstimator, get_array_module


def _soft_ann(X, samples, p):
    xp, xpUtils = get_array_module(X)
    m_dis = []
    for i in range(samples.shape[0]):
        i_sample = samples[i, :]
        l_dis = xp.sum((X - xp.expand_dims(i_sample, 0)) ** p, axis=1) ** (1 / p)
        m_dis.append(l_dis)
    m_dis = xp.array(m_dis) # psi, n
    feature_map = xpUtils.softmax(-m_dis, axis = 0) 
    l_dis_min = xp.sum(m_dis * feature_map, axis=0)
    return xp.transpose(feature_map), l_dis_min


class SoftVoronoiPartitioning(ReservoirSamplingEstimator, TransformerMixin):
    def __init__(self, psi, metric="minkowski", p=2, normalize=False, **kwargs):
        super().__init__(psi)
        self.metric = metric
        self.p = p
        self.normalize = normalize

    def partial_fit(self, X, y=None):
        super().partial_fit(X, y)
        return self

    def transform(self, X):
        xp, _ = get_array_module(X)
        if self.metric == "minkowski" and self.p > 0:
            feature_map, _ = _soft_ann(X, self.samples_, self.p)
            if self.normalize:
                feature_map = feature_map/xp.sum(feature_map)
            return feature_map
        else:
            raise NotImplementedError()

    def score_samples(self, X):
        _, l_dis_min = _soft_ann(X, self.samples_, self.p)
        return -l_dis_min
