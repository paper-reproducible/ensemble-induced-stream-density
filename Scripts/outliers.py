import numpy as np

if __name__ == "__main__":

    # import os
    # import tensorflow as tf

    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
    # tnp = tf.experimental.numpy
    # tnp.experimental_enable_numpy_behavior()
    # xp = tnp
    xp = np

    X = xp.array([[2.1], [3.1], [8.1], [9.1], [100.1]])

    from joblib import Parallel
    from IsolationEstimators import (
        IsolationBasedAnomalyDetector,
        IsolationForestAnomalyDetector,
    )

    np.set_printoptions(precision=2)
    with Parallel(n_jobs=32, prefer="threads") as parallel:

        e = IsolationBasedAnomalyDetector(
            2,
            1000,
            contamination=0.2,
            mass_based=True,
            partitioning_type="fuzzi",
            parallel=parallel,
        )
        result = e.fit_predict(X)
        print("FuzzI by mass: ", result.numpy() if hasattr(result, "numpy") else result)

        e = IsolationBasedAnomalyDetector(
            2,
            1000,
            contamination=0.2,
            mass_based=False,
            partitioning_type="inne",
            parallel=parallel,
        )
        result = e.fit_predict(X)
        print("iNNE by ratio: ", result.numpy() if hasattr(result, "numpy") else result)

        e = IsolationBasedAnomalyDetector(
            2,
            1000,
            contamination=0.2,
            mass_based=True,
            partitioning_type="inne",
            parallel=parallel,
        )
        result = e.fit_predict(X)
        print("iNNE by mass: ", result.numpy() if hasattr(result, "numpy") else result)

        e = IsolationBasedAnomalyDetector(
            2,
            1000,
            contamination=0.2,
            mass_based=True,
            partitioning_type="anne",
            parallel=parallel,
        )
        result = e.fit_predict(X)
        print("aNNE by mass: ", result.numpy() if hasattr(result, "numpy") else result)

        from Common import ball_scale

        X_ = ball_scale(X)

        # e = IsolationBasedAnomalyDetector(
        #     2,
        #     1000,
        #     contamination=0.2,
        #     mass_based=True,
        #     partitioning_type="iforest",
        #     parallel=parallel,
        # )
        # result = e.fit_predict(X_)
        # print(
        #     "iForest by mass: ", result.numpy() if hasattr(result, "numpy") else result
        # )

        e = IsolationForestAnomalyDetector(
            2,
            1000,
            contamination=0.2,
            mass_based=True,
            parallel=parallel,
        )
        result = e.fit_predict(X_)
        print(
            "iForest by mass: ",
            result.numpy() if hasattr(result, "numpy") else result,
        )

        e = IsolationForestAnomalyDetector(
            2,
            1000,
            contamination=0.2,
            mass_based=False,
            parallel=parallel,
        )
        result = e.fit_predict(X_)
        print(
            "iForest by path: ",
            result.numpy() if hasattr(result, "numpy") else result,
        )
