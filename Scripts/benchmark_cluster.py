import os
import numpy as np
import pandas as pd
from time import time
from joblib import Parallel
from sklearn.base import clone
from Common import get_array_module, ball_scale
from Metric import fmeasure
from Clustering import DBSCAN


def estimator(name, eps, minPts, psi, t=1000, parallel=None):
    if name == "fuzzi":
        return lambda: DBSCAN(
            eps=eps, minPts=minPts, metric="fuzzi", psi=psi, t=t, parallel=parallel
        )
    if name == "inne":
        return lambda: DBSCAN(
            eps=eps, minPts=minPts, metric="inne", psi=psi, t=t, parallel=parallel
        )
    if name == "anne":
        return lambda: DBSCAN(
            eps=eps, minPts=minPts, metric="anne", psi=psi, t=t, parallel=parallel
        )
    if name == "iforest":
        return lambda: DBSCAN(
            eps=eps, minPts=minPts, metric="iforest", psi=psi, t=t, parallel=parallel
        )
    # if name == "l2":
    return lambda: DBSCAN(eps=eps, minPts=minPts, psi=psi, t=t, parallel=parallel)


def predict_once(e, X):
    e = e()
    results = e.fit_predict(X)
    print("#", end="")
    return results


def predict_rounds(e, X, y_true, n_rounds=10):
    _, xpUtils = get_array_module(X)
    results = []  # [n_rounds, 1+n]
    metric_results = []
    for round in range(n_rounds):
        seconds = time()
        y_pred = predict_once(e, X)
        seconds = time() - seconds
        y_pred = xpUtils.cast(y_pred, y_true.dtype)
        f1, recall, precision = fmeasure(y_true, y_pred)
        y_pred = xpUtils.to_numpy(y_pred).tolist()
        results.append([round] + y_pred)  # [1, 1+n]
        metric_results.append([f1, recall, precision, seconds])
    metric_results = np.average(metric_results, axis=0).tolist()
    return results, metric_results


def test_params(config, X, y_true, t=1000, n_rounds=10, parallel=None):
    psi_values = config["psi_values"]
    minPts_values = config["minPts_values"]

    test_results = []  # [n_names*n_psis*n_rounds, 3+n]
    test_results_metric = []
    X_ = ball_scale(X)
    for estimator_name in estimator_names:
        eps_values = config["eps_values"][estimator_name]
        for psi in psi_values:
            for minPts in minPts_values:
                for eps in eps_values:
                    e = estimator(estimator_name, eps, minPts, psi, t, parallel)
                    results, metric_results = predict_rounds(
                        e,
                        X_ if estimator_name.startswith("iforest") else X,
                        y_true,
                        n_rounds,
                    )
                    for i in range(len(results)):
                        results[i] = [estimator_name, eps, minPts, psi] + results[i]
                    metric_results = [estimator_name, eps, minPts, psi] + metric_results
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


def save_results(test_results, n_records, file_name):
    df = pd.DataFrame(test_results)
    df.columns = ["detector", "eps", "minPts", "psi", "round"] + [
        "result_" + str(i) for i in range(n_records)
    ]
    save_parquet(df, file_name)
    return


def save_metric(test_results_metric, file_name):
    df = pd.DataFrame(test_results_metric)
    df.columns = [
        "detector",
        "eps",
        "minPts",
        "psi",
        "f1",
        "recall",
        "precision",
        "seconds",
    ]
    save_parquet(df, file_name)
    return


def test(dataset_name, folder="./Data", t=1000, xp=np, parallel=None):
    config = dataset_configs[dataset_name]
    X = config["X"](xp)
    y = config["y"](xp)
    test_results, test_results_metric = test_params(
        config, X, y, t=t, parallel=parallel
    )
    print("\n")
    save_results(test_results, X.shape[0], folder + "/anomaly_pred_" + dataset_name)
    save_metric(test_results_metric, folder + "/anomaly_metric_" + dataset_name)
    return


def main(t=1000, folder="./Data", use_tensorflow=False):
    np.set_printoptions(precision=2)

    if use_tensorflow:
        import tensorflow as tf

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
        tnp = tf.experimental.numpy
        tnp.experimental_enable_numpy_behavior()
        xp = tnp
    else:
        xp = np

    with Parallel(n_jobs=32, prefer="threads") as parallel:
        for dataset_name in dataset_configs:
            test(dataset_name, folder, t=t, xp=xp, parallel=parallel)
    return


estimator_names = ["l2", "anne", "inne", "iforest", "fuzzi"]

dataset_configs = {
    "demo": {
        "X": lambda xp: xp.array([[2.1], [3.1], [8.1], [9.1], [100.1]]),
        "y": lambda xp: xp.array([1, 1, 2, 2, -1]),
        "psi_values": [2],
        "minPts_values": [2],
        "eps_values": {
            "l2": [1.5],
            "anne": [0.2],
            "inne": [0.2],
            "iforest": [0.2],
            "fuzzi": [0.99],
        },
    },
}

_debug = True
_use_tensorflow = False

if __name__ == "__main__":
    main(use_tensorflow=_use_tensorflow)
