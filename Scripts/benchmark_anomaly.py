import numpy as np
import pandas as pd
from time import time
from joblib import Parallel
from sklearn.metrics import precision_recall_fscore_support
from Common import (
    get_array_module,
    ball_scale,
    load_mat,
    save_csv,
    save_parquet,
    init_xp,
)
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
    y_pred = e.fit_predict(X)
    print(".", end="")
    return y_pred


def predict_rounds(e, X, y_true, contamination="auto", n_rounds=10):
    _, xpUtils = get_array_module(X)
    results = []  # [n_rounds, 1+n]
    metric_results = []
    for round in range(n_rounds):
        seconds = time()
        y_pred = predict_once(e, X, contamination)
        seconds = time() - seconds
        y_pred = xpUtils.to_numpy(y_pred).astype(y_true.dtype)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        results.append([round] + y_pred.tolist())  # [1, 1+n]
        metric_results.append([f1, recall, precision, seconds])
    metric_results = np.average(metric_results, axis=0).tolist()
    return results, metric_results


def pred_eval_params(
    X,
    y_true,
    psi_values=[],
    t=1000,
    contamination="auto",
    n_rounds=10,
    parallel=None,
    debug=False,
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
            if debug:
                print("\n", metric_results)
            test_results += results
            test_results_metric += [metric_results]
    return test_results, test_results_metric


def pred_results_to_df(pred_results, n_records):
    df = pd.DataFrame(pred_results)
    df.columns = ["detector", "psi", "round"] + [
        "result_" + str(i) for i in range(n_records)
    ]
    return df


def evaluate_results_to_df(evaluate_results):
    df = pd.DataFrame(evaluate_results)
    df.columns = ["dataset", "detector", "psi", "f1", "recall", "precision", "seconds"]
    return df


def pred_eval(dataset_name, folder="./Data", t=1000, xp=np, parallel=None, debug=False):
    print(" Evaluating dataset " + dataset_name, end="")
    config = dataset_configs[dataset_name]
    X, y = config["data"](xp)
    _, xpUtils = get_array_module(X)
    contamination = config["contamination"]
    psi_values = config["psi_values"]
    pred_results, evaluate_results = pred_eval_params(
        X,
        y_true=xpUtils.to_numpy(y),
        psi_values=psi_values,
        t=t,
        contamination=contamination,
        parallel=parallel,
        debug=debug,
    )
    print(" Done! ")
    df_pred_results = pred_results_to_df(pred_results, X.shape[0])
    if debug:
        print(df_pred_results)
    save_parquet(df_pred_results, folder + "/anomaly_pred_" + dataset_name)
    for i in range(len(evaluate_results)):
        evaluate_results[i] = [dataset_name] + evaluate_results[i]
    return evaluate_results


def main(t=1000, folder="./Data", use_tensorflow=False, use_cupy=False, debug=False):
    np.set_printoptions(precision=2)
    xp = init_xp(use_tensorflow, use_cupy)

    all_evaluate_results = []
    with Parallel(n_jobs=32, prefer="threads") as parallel:
        for dataset_name in dataset_configs:
            evaluate_results = pred_eval(
                dataset_name, folder, t=t, xp=xp, parallel=parallel, debug=debug
            )
            all_evaluate_results += evaluate_results
            if debug:
                print(evaluate_results_to_df(evaluate_results))
    df_all_evaluate_results = evaluate_results_to_df(all_evaluate_results)
    save_csv(df_all_evaluate_results, folder + "/anomaly_metric")
    return


estimator_names = [
    "fuzzi_mass",
    "anne_mass",
    "inne_mass",
    "inne_ratio",
    "iforest_mass",
    "iforest_path",
]

dataset_configs = {
    "demo": {
        "data": lambda xp: (
            xp.array([[2.1], [3.1], [8.1], [9.1], [100.1]]),
            xp.array([1, 1, 1, 1, -1]),
        ),
        "contamination": 0.2,
        "psi_values": [2],
    },
    "http": {
        "data": lambda xp: load_mat("./Data/outlier", "http", xp),
        "contamination": 0.0039,
        "psi_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    },
}

if __name__ == "__main__":
    main(
        use_tensorflow=False,
        use_cupy=False,
        debug=True,
    )
