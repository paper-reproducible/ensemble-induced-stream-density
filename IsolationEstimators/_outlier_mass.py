from joblib import delayed
from Common import get_array_module
from ._outlier_base import BaseAnomalyDetector
from ._data_dependent import MassEstimator
from ._constants import INNE, FUZZI


class IsolationBasedAnomalyDetector(BaseAnomalyDetector):
    def __init__(
        self,
        psi,
        t,
        contamination="auto",
        mass_based=True,
        partitioning_type=INNE,
        n_jobs=16,
        verbose=0,
        parallel=None,
        **kwargs
    ):
        super().__init__(contamination)
        self.mass_estimator = MassEstimator(
            psi,
            t,
            partitioning_type,
            n_jobs,
            verbose,
            parallel,
            anomaly_detection=True,
            **kwargs
        )
        self.partitioning_type = partitioning_type
        self.mass_based = mass_based

    def fit(self, X, y=None):
        self.mass_estimator.fit(X, y)
        return super().fit(X, y)

    def score_samples(self, X):
        if self.mass_based:
            mass = self.mass_estimator.score(X)
            # if self.partitioning_type == FUZZI:
            #     xp, _ = get_array_module(mass)
            #     eps = xp.finfo(mass.dtype).eps
            #     return xp.log(mass + eps)
            return mass
        else:
            xp, _ = get_array_module(X)

            def loop_body(estimator):
                return estimator.score_samples(X)

            all_results = self.mass_estimator.parallel()(
                delayed(loop_body)(i) for i in self.mass_estimator.transformers_
            )
            return xp.average(xp.array(all_results), axis=0)
