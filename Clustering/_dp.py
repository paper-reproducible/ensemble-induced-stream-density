import numpy
from Common import get_array_module


def tree(X, l_rho, metric="minkowski", p=2):
    xp, _ = get_array_module(X)
    n = l_rho.shape[0]
    if X.shape[0] != n:
        raise ValueError("X: [n, :]")

    l_rho_desc = xp.argsort(0 - l_rho)

    l_link = xp.arange(n)
    l_delta = xp.zeros(n)
    if metric == "minkowski" and p > 0:
        for i in range(1, n):
            idx = l_rho_desc[i]
            denser_idx = l_rho_desc[0:i]
            l_dis = xp.sum((X[xp.array([idx]), :] - X[denser_idx, :]) ** p, axis=1)
            l_dis = l_dis ** (1 / p)
            l_link[idx] = denser_idx[xp.argmin(l_dis)]
            l_delta[idx] = xp.min(l_dis)
    else:
        raise NotImplementedError()

    l_gamma = l_delta * l_rho
    return l_link, l_gamma, l_delta


def peaks(l_link, l_gamma, k):
    xp, _ = get_array_module(l_link)
    l_link = xp.copy(l_link)
    n = l_link.shape[0]
    if k > n:
        raise ValueError("k <= n")
    # result = 0 - xp.ones(n)
    l_n = xp.arange(n)
    peaks_mask = l_link == l_n
    peaks_count = xp.sum(peaks_mask)
    # print(l_gamma)
    if peaks_count < k:
        peaks_mask[xp.argsort(l_gamma)[peaks_count - k :]] = True
        l_link[peaks_mask] = l_n[peaks_mask]
        peaks_count = xp.sum(peaks_mask)
    else:
        k = peaks_count

    return peaks_mask, l_link, k


def partition(l_link, peaks_mask, k):
    xp, _ = get_array_module(l_link)
    # l_peaks = xp.expand_dims(xp.where(peaks_mask)[0], axis=0)
    l_peaks = xp.where(peaks_mask)[0]
    l_ancients = xp.expand_dims(l_link, axis=1)
    l_labels = l_link

    while xp.all(xp.any(xp.equal(l_peaks, l_ancients), axis=1)) == False:
        l_labels = l_ancients[:, 0]
        l_labels = l_labels[l_labels]
        l_ancients = xp.expand_dims(l_labels, axis=1)

    for i in range(k):
        l_labels[l_labels == l_peaks[i]] = -i
    l_labels = 0 - l_labels

    return l_labels


from sklearn.base import BaseEstimator, ClusterMixin


class DensityPeak(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_clusters,
        density_estimator,
        n_cache=None,
        cache_peaks=False,
        metric="minkowski",
        p=2,
    ):
        self.n_clusters = n_clusters
        self.density_estimator = density_estimator
        self.n_cache = n_cache
        self.cache_peaks = cache_peaks
        self.metric = metric
        self.p = p
        self.fitted = 0

    def fit(self, X, y=None):
        self.local_cache_ = None
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        xp, _ = get_array_module(X)
        if self.fitted == 0:
            self.density_estimator.fit(X)
        else:
            self.density_estimator.partial_fit(X)

        if self.local_cache_ is None:
            m_test = X
            x_mask = xp.ones(X.shape[0], dtype=bool)
        else:
            m_test = xp.concatenate([self.local_cache_, X], axis=0)
            x_mask = xp.arange(m_test.shape[0]) >= self.local_cache_.shape[0]

        if self.cache_peaks:
            l_rho = self.density_estimator.score(m_test)
            l_link, l_gamma, _ = tree(m_test, l_rho, self.metric, self.p)
            peaks_mask, _, _ = peaks(l_link, l_gamma, self.n_cache)
            self.tmp_link_ = l_link
            self.tmp_gamma_ = l_gamma

            self.local_cache_ = m_test[peaks_mask, :]
            self.cache_x_mask_ = x_mask[peaks_mask, :]
        elif self.n_cache is not None:
            if m_test.shape[0] > self.n_cache:
                self.local_cache_ = m_test[
                    m_test.shape[0] - self.n_cache : m_test.shape[0], :
                ]
                self.cache_x_mask_ = x_mask[
                    m_test.shape[0] - self.n_cache : m_test.shape[0]
                ]
            else:
                self.local_cache_ = m_test
                self.cache_x_mask_ = x_mask

        self.fitted = self.fitted + X.shape[0]
        return self

    def score_density(self, X, segment_size=None):
        xp, _ = get_array_module(X)
        n = X.shape[0]
        if segment_size is None or segment_size >= n:
            return self.density_estimator.score(X)
        else:
            results = []
            for i in range(0, n, segment_size):
                Xi = X[i : i + segment_size, :]
                l_score = self.density_estimator.score(Xi)
                results.append(l_score)
            return xp.concatenate(results, axis=0)

    def predict(self, X, y=None, k=None, segment_size=None):
        if X is None:
            m_test = self.local_cache_
            x_mask = self.cache_x_mask_
            if self.cache_peaks:
                l_link = self.tmp_link_
                l_gamma = self.tmp_gamma_
        else:
            xp, _ = get_array_module(X)
            if self.local_cache_ is None:
                m_test = X
                x_mask = xp.ones(X.shape[0], dtype=bool)
            else:
                m_test = xp.concatenate([self.local_cache_, X], axis=0)
                x_mask = xp.arange(m_test.shape[0]) >= (m_test.shape[0] - X.shape[0])

        if X is not None or not self.cache_peaks:
            # l_rho = self.density_estimator.score(m_test)
            l_rho = self.score_density(m_test, segment_size=segment_size)
            l_link, l_gamma, _ = tree(m_test, l_rho, self.metric, self.p)

        if k == None:
            k = self.n_clusters
        peaks_mask, l_link, k = peaks(l_link, l_gamma, k)
        mk, _ = get_array_module(k)
        if mk != numpy:
            k = k.get()
        l_labels_full = partition(l_link, peaks_mask, k)
        self.tmp_all_lables_ = l_labels_full
        return l_labels_full[x_mask]

    def fit_predict(self, X, y=None, k=None, segment_size=None):
        self.n_cache = X.shape[0]
        self.local_cache_ = None
        return self.partial_fit(X, y).predict(None, k=k, segment_size=segment_size)
