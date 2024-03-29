from abc import abstractmethod
from sklearn.base import BaseEstimator, clone
from joblib import Parallel, delayed
from Common import get_array_module


class BaseBaggingEstimator(BaseEstimator):
    def __init__(
        self, partitioning_initializer, t, n_jobs=16, verbose=0, parallel=None, **kwargs
    ):
        self.partitioning_initializer = partitioning_initializer
        self.t = t
        self.n_jobs = n_jobs
        self.verbose = verbose
        # self.transformers_ = [partitioning_initializer() for _ in range(t)]
        self.fitted = 0
        self.preset_parallel = parallel

    def parallel(self):
        if self.preset_parallel is None:
            return Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads")
        else:
            return self.preset_parallel

    @abstractmethod
    def fit(self, X, y=None):
        def single_fit(bagger, X):
            e = bagger.partitioning_initializer()
            return e

        self.transformers_ = self.parallel()(
            delayed(single_fit)(self, X) for _ in range(self.t)
        )
        self.fitted = X.shape[0]
        return self


class BaseAdaptiveBaggingEstimator(BaseBaggingEstimator):
    def partial_fit(self, X, y=None):
        if self.fitted == 0:
            return self.fit(X, y)

        self.transformers_ = self.parallel()(
            delayed(i.partial_fit)(X) for i in self.transformers_
        )

        self.fitted = self.fitted + X.shape[0]
        return self
