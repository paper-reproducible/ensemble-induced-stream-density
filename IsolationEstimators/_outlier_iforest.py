import numpy as np
from joblib import delayed
from Common import get_array_module
from ._outlier_base import BaseAnomalyDetector
from ._data_dependent import IncrementalMassEstimator, IsolationTransformer
from ._constants import IFOREST
from ._isolation_tree import IsolationTree


class IsolationForestAnomalyDetector(BaseAnomalyDetector):
    def __init__(
        self,
        psi,
        t,
        contamination="auto",
        mass_based=True,
        n_jobs=16,
        verbose=0,
        parallel=None,
        **kwargs
    ):
        super().__init__(contamination)
        if mass_based:
            self.iforest = IncrementalMassEstimator(
                psi,
                t,
                partitioning_type=IFOREST,
                n_jobs=n_jobs,
                verbose=verbose,
                parallel=parallel,
                **kwargs
            )
        else:
            self.iforest = IsolationTransformer(
                psi,
                t,
                partitioning_type=IFOREST,
                n_jobs=n_jobs,
                verbose=verbose,
                parallel=parallel,
                **kwargs
            )
        self.mass_based = mass_based

    def fit(self, X, y=None):
        self.iforest.fit(X, y)
        # xp, _ = get_array_module(X)
        # for tree in self.iforest.transformers_:
        #     tree.backup = xp.copy(tree.combine_boundaries())
        return super().fit(X, y)

    def score_samples(self, X):
        if self.mass_based:
            return self.iforest.score(X)
        else:
            xp, _ = get_array_module(X)
            n, _ = X.shape

            def loop_body(tree: IsolationTree):
                indices = tree.transform(X)  # [n, 1] for itree
                # if hasattr(tree, "backup"):
                #     boundaries = tree.combine_boundaries()
                #     print(xp.all(boundaries == tree.backup))
                indices = xp.squeeze(indices, axis=1)
                return xp.take(tree.node_levels[tree.node_is_leaf_], indices, axis=0)

            levels = self.iforest.parallel()(
                delayed(loop_body)(i) for i in self.iforest.transformers_
            )
            avg_path = xp.average(xp.array(levels), axis=0)
            H = lambda i: xp.log(i) + np.euler_gamma
            c = lambda n: 2 * H(n - 1) - (2 * (n - 1) / n)
            anomaly_score = 2 ** (avg_path / c(n))
            return anomaly_score
