import os
import numpy as np
import pandas as pd
from time import time
from joblib import Parallel
from Common import get_array_module, ball_scale
from Metric import fmeasure
from IsolationEstimators import (
    IsolationBasedAnomalyDetector,
    IsolationForestAnomalyDetector,
)


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
    results = e.fit_predict(X)
    print("#", end="")
    return results


def predict_rounds(e, X, y_true, contamination="auto", n_rounds=10):
    _, xpUtils = get_array_module(X)
    results = []  # [n_rounds, 1+n]
    metric_results = []
    for round in range(n_rounds):
        seconds = time()
        y_pred = predict_once(e, X, contamination)
        seconds = time() - seconds
        y_pred = xpUtils.cast(y_pred, y_true.dtype)
        f1, recall, precision = fmeasure(y_true, y_pred)
        y_pred = xpUtils.to_numpy(y_pred).tolist()
        results.append([round] + y_pred)  # [1, 1+n]
        metric_results.append([f1, recall, precision, seconds])
    metric_results = np.average(metric_results, axis=0).tolist()
    return results, metric_results


def test_params(
    X, y_true, psi_values=[], t=1000, contamination="auto", n_rounds=10, parallel=None
):
    test_results = []  # [n_names*n_psis*n_rounds, 3+n]
    test_results_metric = []
    X_ = ball_scale(X)
    for estimator_name in estimator_names:
        for psi in psi_values:
            e = estimator(estimator_name, psi, t, parallel)
            results, metric_results = predict_rounds(
                e,
                X_ if estimator_name.startswith("iforest") else X,
                y_true,
                contamination,
                n_rounds,
            )
            for i in range(len(results)):
                results[i] = [estimator_name, psi] + results[i]
            metric_results = [estimator_name, psi] + metric_results
            test_results += results
            test_results_metric += [metric_results]
    return test_results, test_results_metric


def save_parquet(df, file_name):
    ext = ".parquet.gzip"
    file_name = file_name if file_name.endswith(ext) else file_name + ext
    if _debug:
        print("Saving " + file_name + ": ")
        print(df)
    df.to_parquet(file_name, compression="gzip")
    return
    

def save_csv(df, file_name):
    ext = ".csv"
    file_name = file_name if file_name.endswith(ext) else file_name + ext
    df.to_csv(file_name)


def save_results(test_results, n_records, file_name):
    df = pd.DataFrame(test_results)
    df.columns = ["detector", "psi", "round"] + [
        "result_" + str(i) for i in range(n_records)
    ]
    save_parquet(df, file_name)
    return


def save_metric(test_results_metric, file_name):
    df = pd.DataFrame(test_results_metric)
    df.columns = ["dataset", "detector", "psi", "f1", "recall", "precision", "seconds"]
    save_csv(df, file_name)
    return


def test(dataset_name, folder="./Data", t=1000, xp=np, parallel=None):
    config = dataset_configs[dataset_name]
    X = config["X"](xp)
    y = config["y"](xp)
    contamination = config["contamination"]
    psi_values = config["psi_values"]
    test_results, test_results_metric = test_params(
        X, y, psi_values, t=t, contamination=contamination, parallel=parallel
    )
    print("\n")
    save_results(test_results, X.shape[0], folder + "/anomaly_pred_" + dataset_name)
    for i in range(len(test_results_metric)):
        test_results_metric[i] = [dataset_name]+test_results_metric[i]
    return test_results_metric


def main(t=1000, folder="./Data", use_tensorflow=False, use_cupy=False):
    np.set_printoptions(precision=2)

    if use_tensorflow:
        import tensorflow as tf

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
        tnp = tf.experimental.numpy
        tnp.experimental_enable_numpy_behavior()
        xp = tnp
    elif use_cupy:
        import cupy as cp
        xp = cp
    else:
        xp = np

    test_results_metric = []
    with Parallel(n_jobs=32, prefer="threads") as parallel:
        for dataset_name in dataset_configs:
            test_results_metric += test(dataset_name, folder, t=t, xp=xp, parallel=parallel)
    save_metric(test_results_metric, folder + "/anomaly_metric")
    return


estimator_names = [
    "inne_ratio",
    "inne_mass",
    "fuzzi_mass",
    "anne_mass",
    # "iforest_mass",
    # "iforest_path",
]

dataset_configs = {
    "demo": {
        "X": lambda xp: xp.array([[2.1], [3.1], [8.1], [9.1], [100.1]]),
        "y": lambda xp: xp.array([1, 1, 1, 1, -1]),
        "contamination": 0.2,
        "psi_values": [2],
    },
}

_debug = True
_use_tensorflow = False
_use_cupy = False

if __name__ == "__main__":
    main(
        use_tensorflow=_use_tensorflow,
        use_cupy=_use_cupy,
    )
