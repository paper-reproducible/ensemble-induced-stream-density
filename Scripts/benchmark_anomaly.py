import numpy as np
import pandas as pd
import h5py
import scipy.io
from time import time
from joblib import Parallel
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
    all_label_results = []  # [n_names*n_psis*n_rounds, 3+n]
    all_score_results = []  # [n_names*n_psis*n_rounds, 3+n]
    all_metric_results = []
    X_ = ball_scale(X)
    for estimator_name in estimator_names:
        for psi in psi_values:
            e = estimator(estimator_name, psi, t, parallel)
            label_results, score_results, metric_results = predict_rounds(
                e,
                X_ if estimator_name.startswith("iforest") else X,
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
    np.set_printoptions(precision=2, linewidth=150)
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


# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html
def preprocess_dataset(dataset_name):
    from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_openml
    from sklearn.preprocessing import LabelBinarizer

    rng = np.random.RandomState(42)

    # loading and vectorization
    # print(f"Loading {dataset_name} data")
    if dataset_name in ["http", "smtp", "SA", "SF"]:
        dataset = fetch_kddcup99(subset=dataset_name, percent10=True, random_state=rng)
        X = dataset.data
        y = dataset.target
        lb = LabelBinarizer()

        if dataset_name == "SF":
            idx = rng.choice(X.shape[0], int(X.shape[0] * 0.1), replace=False)
            X = X[idx]  # reduce the sample size
            y = y[idx]
            x1 = lb.fit_transform(X[:, 1].astype(str))
            X = np.c_[X[:, :1], x1, X[:, 2:]]
        elif dataset_name == "SA":
            idx = rng.choice(X.shape[0], int(X.shape[0] * 0.1), replace=False)
            X = X[idx]  # reduce the sample size
            y = y[idx]
            x1 = lb.fit_transform(X[:, 1].astype(str))
            x2 = lb.fit_transform(X[:, 2].astype(str))
            x3 = lb.fit_transform(X[:, 3].astype(str))
            X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != b"normal.").astype(int)
    if dataset_name == "forestcover":
        dataset = fetch_covtype()
        X = dataset.data
        y = dataset.target
        idx = rng.choice(X.shape[0], int(X.shape[0] * 0.1), replace=False)
        X = X[idx]  # reduce the sample size
        y = y[idx]

        # inliers are those with attribute 2
        # outliers are those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)
    if dataset_name in ["glass", "wdbc", "cardiotocography"]:
        dataset = fetch_openml(name=dataset_name, version=1, as_frame=False)
        X = dataset.data
        y = dataset.target

        if dataset_name == "glass":
            s = y == "tableware"
            y = s.astype(int)
        if dataset_name == "wdbc":
            s = y == "2"
            y = s.astype(int)
            X_mal, y_mal = X[s], y[s]
            X_ben, y_ben = X[~s], y[~s]

            # downsampled to 39 points (9.8% outliers)
            idx = rng.choice(y_mal.shape[0], 39, replace=False)
            X_mal2 = X_mal[idx]
            y_mal2 = y_mal[idx]
            X = np.concatenate((X_ben, X_mal2), axis=0)
            y = np.concatenate((y_ben, y_mal2), axis=0)
        if dataset_name == "cardiotocography":
            s = y == "3"
            y = s.astype(int)
    # 0 represents inliers, and 1 represents outliers
    y = pd.Series(y, dtype="category")
    return (X, y)


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


def load_odds(folder, dataset_name, xp=np):
    file_name = folder + "/" + dataset_name + ".mat"
    try:
        f = h5py.File(file_name, "r")
    except:
        f = scipy.io.loadmat(file_name)
    X = xp.array(f.get("X"))
    y = xp.array(f.get("y"))
    if len(y.shape) == 2:
        if y.shape[1] == 1:
            y = xp.squeeze(y, axis=1)
        elif y.shape[0] == 1:
            y = xp.squeeze(y, axis=0)
    if X.shape[1] == y.shape[0]:
        X = xp.transpose(X)
    labels = xp.sort(xp.unique(y))
    if labels.shape[0] == 2 and labels[0] == 0 and labels[1] == 1:
        y = xp.where(y == 1, -1, y)
        y = xp.where(y == 0, 1, y)
    elif xp.min(labels) < 0:
        y = xp.where(y >= 0, 1, y)
        y = xp.where(y < 0, -1, y)
    else:
        raise Exception("Unsupported label format")
    return X, y


def load_sklearn(dataset_name, xp=np):
    X, y = preprocess_dataset(dataset_name)
    X = xp.array(X, dtype=float)
    y = xp.array(y, dtype=int)
    # For sklearn data
    y = xp.where(y == 1, -1, y)
    y = xp.where(y == 0, 1, y)
    return X, y


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
        "data": lambda xp: load_sklearn("http", xp),
        "contamination": 2209.0 / (2209.0 + 56516.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "smtp": {
        "data": lambda xp: load_sklearn("smtp", xp),
        "contamination": 3.0 / (3.0 + 9568.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "SA": {
        "data": lambda xp: load_sklearn("SA", xp),
        "contamination": 344.0 / (344.0 + 9721.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "SF": {
        "data": lambda xp: load_sklearn("SF", xp),
        "contamination": 320.0 / (320.0 + 7003.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "forestcover": {
        "data": lambda xp: load_sklearn("forestcover", xp),
        "contamination": 259.0 / (259.0 + 28248.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "glass": {
        "data": lambda xp: load_sklearn("glass", xp),
        "contamination": 9.0 / (9.0 + 205.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "wdbc": {
        "data": lambda xp: load_sklearn("wdbc", xp),
        "contamination": 39.0 / (39.0 + 357.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "cardiotocography": {
        "data": lambda xp: load_sklearn("cardiotocography", xp),
        "contamination": 53.0 / (53.0 + 2073.0),
        "psi_values": [2, 4, 8, 16, 32],
    },
    "mnist": {
        "data": lambda xp: load_odds("./Data/odds", "mnist", xp),
        "contamination": 700.0 / 7603.0,
        "psi_values": [2, 4, 8, 16, 32],
    },
}

if __name__ == "__main__":
    main(
        use_tensorflow=False,
        use_cupy=True,
        debug=True,
    )
