from sklearn.base import BaseEstimator, clone
from joblib import Parallel, delayed


class BaseBaggingEstimator(BaseEstimator):
    def __init__(self, base_transformer, t, n_jobs=16, verbose=0, parallel=None):
        self.base_transformer = base_transformer
        self.t = t
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.transformers_ = [clone(base_transformer) for _ in range(t)]
        self.fitted = 0
        self.preset_parallel = parallel

    def parallel(self):
        if self.preset_parallel is None:
            return Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads"
            )
        else:
            return self.preset_parallel

    def fit(self, X, y=None):
        self.parallel()(delayed(i.fit)(X) for i in self.transformers_)

        self.fitted = X.shape[0]
        return self


class BaseAdaptiveBaggingEstimator(BaseBaggingEstimator):
    def partial_fit(self, X, y=None):
        if self.fitted == 0:
            return self.fit(X, y)

        self.parallel()(delayed(i.partial_fit)(X) for i in self.transformers_)

        self.fitted = self.fitted + X.shape[0]
        return self
