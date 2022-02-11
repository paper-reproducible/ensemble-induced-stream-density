from joblib import Parallel, delayed
from sklearn.base import DensityMixin, BaseEstimator
from sklearn.utils.fixes import _joblib_parallel_args
from Common import ReservoirSamplingEstimator, get_array_module


def gaussian(X):
    xp, _ = get_array_module(X)
    return xp.exp(-0.5 * xp.sum(X ** 2, axis=1))


def kde(X, Y, bw, kernel_function):
    xp, _ = get_array_module(X)
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    X_ = xp.expand_dims(X, 1)
    Y_ = xp.expand_dims(Y, 0)

    X_Y = (X_ - Y_) / bw

    K_X_Y = kernel_function(X_Y.reshape([n_X * n_Y, -1]))
    K_X_Y = K_X_Y.reshape([n_X, n_Y])

    return K_X_Y.sum(axis=1) / (n_Y * bw)


KERNEL_GAUSSIAN = "gaussian"


class SingleKDE(ReservoirSamplingEstimator, DensityMixin):
    def __init__(self, psi, bandwidth, kernel=KERNEL_GAUSSIAN):
        super().__init__(psi)
        self.bandwidth = bandwidth
        self.kernel = kernel
        if kernel != KERNEL_GAUSSIAN:
            raise NotImplementedError()

    def score(self, X, y=None):
        if self.kernel == KERNEL_GAUSSIAN:
            return kde(X, self.samples_, bw=self.bandwidth, kernel_function=gaussian)
        else:
            raise NotImplementedError()


class AdaptiveKernelDensityEstimator(BaseEstimator, DensityMixin):
    def __init__(
        self,
        psi,
        t,
        bandwidth,
        kernel=KERNEL_GAUSSIAN,
        n_jobs=16,
        verbose=0,
        parallel=None,
    ):

        self.estimators = [SingleKDE(psi, bandwidth, kernel) for _ in range(t)]
        self.psi = psi
        self.t = t
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.fitted = 0
        self.preset_parallel = parallel

    def parallel(self):
        if self.preset_parallel is None:
            return Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                **_joblib_parallel_args(prefer="threads")
            )
        else:
            return self.preset_parallel

    def fit(self, X, y=None):
        self.parallel()(delayed(i.fit)(X) for i in self.estimators)

        self.fitted = X.shape[0]
        return self

    def partial_fit(self, X, y=None):
        self.parallel()(delayed(i.partial_fit)(X) for i in self.estimators)

        self.fitted = self.fitted + X.shape[0]
        return self

    def score(self, X, y=None):
        xp, _ = get_array_module(X)

        def loop_body(estimator):
            return estimator.score(X)

        all_results = Parallel()(delayed(loop_body)(i) for i in self.estimators)
        return xp.average(xp.array(all_results), axis=0)
