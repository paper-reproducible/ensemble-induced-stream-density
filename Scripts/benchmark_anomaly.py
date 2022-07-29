import os
import numpy as np
import pandas as pd
from joblib import Parallel
from sklearn.base import clone
from Common import get_array_module, ball_scale
from IsolationEstimators import (
    IsolationBasedAnomalyDetector,
    IsolationForestAnomalyDetector,
)

estimator_names = [
    "fuzzi_mass",
    "inne_mass",
    "anne_mass",
    "iforest_mass",
    "inne_ratio",
    "iforest_path",
]


def estimator(name, psi, t=1000, parallel=None):
    if name == "fuzzi_mass":
        return lambda: IsolationBasedAnomalyDetector(
            psi,
            t,
            mass_based=True,
            partitioning_type="fuzzi",
            parallel=parallel,
        )
    if name == "inne_mass":
        return lambda: IsolationBasedAnomalyDetector(
            psi,
            t,
            mass_based=True,
            partitioning_type="inne",
            parallel=parallel,
        )
    if name == "inne_ratio":
        return lambda: IsolationBasedAnomalyDetector(
            psi,
            t,
            mass_based=False,
            partitioning_type="inne",
            parallel=parallel,
        )
    if name == "iforest_path":
        return lambda: IsolationForestAnomalyDetector(
            psi,
            t,
            contamination=0.2,
            mass_based=False,
            parallel=parallel,
        )
    if name == "iforest_mass":
        return lambda: IsolationForestAnomalyDetector(
            psi,
            t,
            mass_based=True,
            parallel=parallel,
        )
    if name == "anne_mass":
        return lambda: IsolationBasedAnomalyDetector(
            psi,
            t,
            mass_based=True,
            partitioning_type="anne",
            parallel=parallel,
        )


def predict_once(e, X, contamination="auto"):
    e = e()
    e.contamination = contamination
    return e.fit_predict(X)


def predict_rounds(e, X, y, contamination="auto", n_rounds=10):
    _, xpUtils = get_array_module(X)

    results = []  # [n_rounds, 1+n]
    for round in range(n_rounds):
        y_pred = predict_once(e, X, contamination)
        y_pred = xpUtils.to_numpy(y_pred).tolist()
        results.append([round] + y_pred)  # [1, 1+n]

    return results


def test_params(
    X, y, psi_values=[], t=1000, contamination="auto", n_rounds=10, parallel=None
):
    test_results = []  # [n_names*n_psis*n_rounds, 3+n]
    X_ = ball_scale(X)
    for estimator_name in estimator_names:
        for psi in psi_values:
            e = estimator(estimator_name, psi, t, parallel)
            results = predict_rounds(
                e,
                X_ if estimator_name.startswith("iforest") else X,
                y,
                contamination,
                n_rounds,
            )
            for i in range(len(results)):
                results[i] = [estimator_name, psi] + results[i]
            test_results += results
    return test_results


def save_parquet(df, file_name):
    if file_name.endswith("demo"):
        print(df)  # test
    ext = ".parquet.gzip"
    file_name = file_name if file_name.endswith(ext) else file_name + ext
    df.to_parquet(file_name, compression="gzip")
    return


def save_results(test_results, n_records, file_name):
    df = pd.DataFrame(test_results)
    df.columns = ["detector", "psi", "round"] + [
        "result_" + str(i) for i in range(n_records)
    ]
    save_parquet(df, file_name)
    return


dataset_configs = {
    "demo": {
        "X": lambda xp: xp.array([[2.1], [3.1], [8.1], [9.1], [100.1]]),
        "y": lambda xp: xp.array([1, 1, 1, 1, -1]),
        "contamination": 0.2,
        "psi_values": [2],
    },
}


def test(dataset_name, t=1000, xp=np, parallel=None):
    config = dataset_configs[dataset_name]
    X = config["X"](xp)
    y = config["y"](xp)
    contamination = config["contamination"]
    psi_values = config["psi_values"]
    test_results = test_params(
        X, y, psi_values, t=t, contamination=contamination, parallel=parallel
    )
    save_results(test_results, X.shape[0], "./Data/anomaly_test_" + dataset_name)
    return


def main(t=1000, use_tensorflow=False):
    np.set_printoptions(precision=2)

    if use_tensorflow:
        import tensorflow as tf

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
        tnp = tf.experimental.numpy
        tnp.experimental_enable_numpy_behavior()
        xp = tnp
    else:
        xp = np

    with Parallel(n_jobs=1, prefer="threads") as parallel:
        for dataset_name in dataset_configs:
            test(dataset_name, t=t, xp=xp, parallel=parallel)

    return


if __name__ == "__main__":
    main(use_tensorflow=False)
