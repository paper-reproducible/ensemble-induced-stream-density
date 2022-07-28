from sklearn.base import DensityMixin
from joblib import delayed
from ._outlier_base import BaseAnomalyDetector
from ._data_dependent import MassEstimator
from Common import get_array_module
from ._constants import INNE


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
        if hasattr(self.mass_estimator.base_transformer, "score_samples"):
            self.mass_based = mass_based
        else:
            self.mass_based = True

    def fit(self, X, y=None):
        self.mass_estimator.fit(X, y)
        return super().fit(X, y)

    def score_samples(self, X):
        if self.mass_based:
            return self.mass_estimator.score(X)
        else:
            xp, _ = get_array_module(X)

            def loop_body(estimator):
                return estimator.score_samples(X)

            all_results = self.mass_estimator.parallel()(
                delayed(loop_body)(i) for i in self.mass_estimator.transformers_
            )
            return xp.average(xp.array(all_results), axis=0)
