import numpy as np
import pandas as pd
from time import time
from joblib import Parallel
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from Common import (
    get_array_module,
    ball_scale,
    save_csv,
    save_parquet,
    init_xp,
)
from IsolationEstimators import (
    IsolationBasedAnomalyDetector,
    IsolationForestAnomalyDetector,
)
from Data._outliers import load_odds, load_sklearn_real, load_sklearn_artificial


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
    if name == "iforest_sklearn":
        return lambda: IsolationForest(
            max_samples=psi,
            n_estimators=t,
            n_jobs=32,
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
    if name == "anne_dis":
        return lambda: IsolationBasedAnomalyDetector(
            psi,
            t,
            mass_based=False,
            partitioning_type="anne",
            parallel=parallel,
        )
    if name == "soft_anne_mass":
        return lambda: IsolationBasedAnomalyDetector(
            psi,
            t,
            mass_based=True,
            partitioning_type="soft_anne",
            parallel=parallel,
        )
    if name == "soft_anne_dis":
        return lambda: IsolationBasedAnomalyDetector(
            psi,
            t,
            mass_based=False,
            partitioning_type="soft_anne",
            parallel=parallel,
        )


def predict_once(e, X, contamination="auto"):
    e = e()
    e.contamination = contamination
    y_pred = e.fit_predict(X)
    y_score = e.decision_function(X)
    print(".", end="")
    return y_pred, y_score


def predict_rounds(e, X, y_true, contamination="auto", n_rounds=10):
    _, xpUtils = get_array_module(X)
    label_results = []  # [n_rounds, 1+n]
    score_results = []
    metric_results = []
    for round in range(n_rounds):
        seconds = time()
        y_pred, y_score = predict_once(e, X, contamination)
        seconds = time() - seconds
        y_pred = xpUtils.to_numpy(y_pred).astype(y_true.dtype)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        y_score = xpUtils.to_numpy(y_score)
        roc = roc_auc_score(y_true, y_score)
        label_results.append([round] + y_pred.tolist())  # [1, 1+n]
        score_results.append([round] + y_score.tolist())  # [1, 1+n]
        metric_results.append([f1, recall, precision, roc, seconds])
    metric_avg = np.average(metric_results, axis=0).tolist()
    metric_std = np.std(metric_results, axis=0).tolist()
    return label_results, score_results, metric_avg + metric_std


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
    _, xpUtils = get_array_module(X)
    all_label_results = []  # [n_names*n_psis*n_rounds, 3+n]
    all_score_results = []  # [n_names*n_psis*n_rounds, 3+n]
    all_metric_results = []
    X_ = ball_scale(X)
    for estimator_name in estimator_names:
        for psi in psi_values:
            e = estimator(estimator_name, psi, t, parallel)
            label_results, score_results, metric_results = predict_rounds(
                e,
                xpUtils.to_numpy(X)
                if estimator_name == "iforest_sklearn"
                else X_
                if estimator_name.startswith("iforest")
                else X,
                y_true,
                contamination,
                n_rounds,
            )
            for i in range(len(label_results)):
                label_results[i] = [estimator_name, psi] + label_results[i]
                score_results[i] = [estimator_name, psi] + score_results[i]
            metric_results = [estimator_name, psi] + metric_results
            if debug:
                print(
                    np.array(metric_results[2:]),
                    "psi=" + str(metric_results[1]),
                    "using " + metric_results[0],
                )
            all_label_results += label_results
            all_score_results += score_results
            all_metric_results += [metric_results]
    return all_label_results, all_score_results, all_metric_results


def pred_results_to_df(pred_results, n_records):
    df = pd.DataFrame(pred_results)
    df.columns = ["detector", "psi", "round"] + [
        "result_" + str(i) for i in range(n_records)
    ]
    return df


def evaluate_results_to_df(evaluate_results):
    df = pd.DataFrame(evaluate_results)
    df.columns = [
        "dataset",
        "detector",
        "psi",
        "f1",
        "recall",
        "precision",
        "roc_auc",
        "seconds",
        "f1_std",
        "recall_std",
        "precision_std",
        "roc_auc_std",
        "seconds_std",
    ]
    return df


def find_best_psi_by_roc_auc(evaluate_results):
    psi_map = {}
    roc_auc_map = {}
    for result in evaluate_results:
        estimator_name = result[0]
        psi = result[1]
        score = result[5]
        if estimator_name not in roc_auc_map or roc_auc_map[estimator_name] < score:
            roc_auc_map[estimator_name] = score
            psi_map[estimator_name] = psi
    return psi_map


def pred_eval(dataset_name, folder="./Data", t=1000, xp=np, parallel=None, debug=False):
    print(" Evaluating dataset " + dataset_name + ":")
    config = dataset_configs[dataset_name]
    X, y = config["data"](xp)
    _, xpUtils = get_array_module(X)
    contamination = config["contamination"]
    psi_values = config["psi_values"]
    label_results, score_results, evaluate_results = pred_eval_params(
        X,
        y_true=xpUtils.to_numpy(y),
        psi_values=psi_values,
        t=t,
        contamination=contamination,
        parallel=parallel,
        debug=debug,
    )
    print(" Done! ")

    best_psi_map = find_best_psi_by_roc_auc(evaluate_results)

    estimator_y_score = {}
    for score_result in score_results:
        estimator_name = score_result[0]
        if score_result[1] == best_psi_map[estimator_name] and score_result[2] == 0:
            estimator_y_score[estimator_name] = score_result[3:]

    fig, _ = plot_roc(
        dataset_name, xpUtils.to_numpy(y), estimator_y_score, show=False, block=False
    )
    fig.savefig(folder + "/anomaly_roc_" + dataset_name + ".eps", format="eps")

    df_label_results = pred_results_to_df(label_results, X.shape[0])
    if debug:
        print(df_label_results)
    save_parquet(df_label_results, folder + "/anomaly_label_" + dataset_name)

    df_score_results = pred_results_to_df(score_results, X.shape[0])
    if debug:
        print(df_score_results)
    save_parquet(df_score_results, folder + "/anomaly_score_" + dataset_name)

    for i in range(len(evaluate_results)):
        evaluate_results[i] = [dataset_name] + evaluate_results[i]
    return evaluate_results


def main(t=1000, folder="./Data", use_tensorflow=False, use_cupy=False, debug=False):
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
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


def plot_roc(
    dataset_name, y_true, estimator_y_score, show=True, block=True, fig=None, ax=None
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.axes()

    linewidth = 1
    pos_label = 1  # mean 1 belongs to positive class

    for estimator_name in estimator_y_score:
        y_score = np.array(estimator_y_score[estimator_name])
        display = RocCurveDisplay.from_predictions(
            y_true,
            y_score,
            pos_label=pos_label,
            name=estimator_name,
            linewidth=linewidth,
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], linewidth=linewidth, linestyle=":")
    ax.set_title(dataset_name)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    if show:
        plt.show(block=block)

    return fig, ax


estimator_names = [
    "fuzzi_mass",
    "soft_anne_mass",
    "soft_anne_dis",
    "inne_mass",
    "inne_ratio",
    "anne_mass",
    "anne_dis",
    "iforest_sklearn",
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
        "data": lambda xp: load_sklearn_real("http", xp),
        "contamination": 2209.0 / (2209.0 + 56516.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "smtp": {
        "data": lambda xp: load_sklearn_real("smtp", xp),
        "contamination": 3.0 / (3.0 + 9568.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "SA": {
        "data": lambda xp: load_sklearn_real("SA", xp),
        "contamination": 344.0 / (344.0 + 9721.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "SF": {
        "data": lambda xp: load_sklearn_real("SF", xp),
        "contamination": 320.0 / (320.0 + 7003.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "forestcover": {
        "data": lambda xp: load_sklearn_real("forestcover", xp),
        "contamination": 259.0 / (259.0 + 28248.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "glass": {
        "data": lambda xp: load_sklearn_real("glass", xp),
        "contamination": 9.0 / (9.0 + 205.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "wdbc": {
        "data": lambda xp: load_sklearn_real("wdbc", xp),
        "contamination": 39.0 / (39.0 + 357.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "cardiotocography": {
        "data": lambda xp: load_sklearn_real("cardiotocography", xp),
        "contamination": 53.0 / (53.0 + 2073.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "mnist": {
        "data": lambda xp: load_odds("./Data/odds", "mnist", xp),
        "contamination": 700.0 / 7603.0,
        "psi_values": [2, 4, 8, 16, 32],
    },
    "single_blob": {
        "data": lambda xp: load_sklearn_artificial("one_blob", xp),
        "contamination": 0.15,
        "psi_values": [2, 4, 8, 16, 32],
    },
    "two_blobs": {
        "data": lambda xp: load_sklearn_artificial("two_blobs", xp),
        "contamination": 0.15,
        "psi_values": [2, 4, 8, 16, 32],
    },
    "imba_blobs": {
        "data": lambda xp: load_sklearn_artificial("imbalanced", xp),
        "contamination": 0.15,
        "psi_values": [2, 4, 8, 16, 32],
    },
    "two_moons": {
        "data": lambda xp: load_sklearn_artificial("two_moons", xp),
        "contamination": 0.15,
        "psi_values": [2, 4, 8, 16, 32],
    },
    ## ROC_AUC does not work as th
    # "uniform": {
    #     "data": lambda xp: load_sklearn_artificial("uniform", xp),
    #     "contamination": 0.15,
    #     "psi_values": [2, 4, 8, 16, 32],
    # },
}

if __name__ == "__main__":
    main(
        use_tensorflow=False,
        use_cupy=False,
        debug=True,
    )
