import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from joblib import Parallel
from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from IsolationEstimators import (
    IsolationBasedAnomalyDetector,
    IsolationForestAnomalyDetector,
)

# from sklearn.metrics import roc_auc_score
from Data._outliers import load_sklearn_artificial
from Common import ball_scale


dataset_configs = {
    "single_blob": {
        "data": lambda xp=np: load_sklearn_artificial("one_blob", xp),
        "contamination": 0.15,
        "psi": 2,
    },
    "two_blobs": {
        "data": lambda xp=np: load_sklearn_artificial("two_blobs", xp),
        "contamination": 0.15,
        "psi": 32,
    },
    "imba_blobs": {
        "data": lambda xp=np: load_sklearn_artificial("imbalanced", xp),
        "contamination": 0.15,
        "psi": 32,
    },
    "two_moons": {
        "data": lambda xp=np: load_sklearn_artificial("two_moons", xp),
        "contamination": 45.0 / 345,
        "psi": 16,
    },
    "uniform": {
        "data": lambda xp=np: load_sklearn_artificial("uniform", xp),
        "contamination": 0.15,
        "psi": 2,
    },
}

_t = 1000
_n_jobs = 32

estimator_configs = {
    "fuzzi_mass": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=True,
        partitioning_type="fuzzi",
        parallel=parallel,
    ),
    "inne_mass": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=True,
        partitioning_type="inne",
        parallel=parallel,
    ),
    "inne_ratio": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=False,
        partitioning_type="inne",
        parallel=parallel,
    ),
    "anne_mass": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=True,
        partitioning_type="anne",
        parallel=parallel,
    ),
    "anne_dis": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=False,
        partitioning_type="anne",
        parallel=parallel,
    ),
    "soft_anne_mass": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=True,
        partitioning_type="soft_anne",
        parallel=parallel,
    ),
    "soft_anne_dis": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=False,
        partitioning_type="soft_anne",
        parallel=parallel,
    ),
    "iforest_sklearn": lambda psi, contamination, parallel: IsolationForest(
        max_samples=psi,
        n_estimators=_t,
        contamination=contamination,
        n_jobs=_n_jobs,
    ),
    "iforest_path": lambda psi, contamination, parallel: IsolationForestAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=False,
        parallel=parallel,
    ),
    "iforest_mass": lambda psi, contamination, parallel: IsolationForestAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=True,
        parallel=parallel,
    ),
}

matplotlib.rcParams["contour.negative_linestyle"] = "solid"


# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
xxyy = np.c_[xx.ravel(), yy.ravel()]
xxyy_ = ball_scale(xxyy)

plt.figure(figsize=(len(estimator_configs) * 2 + 4, 12.5))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
)

plot_num = 1

with Parallel(n_jobs=_n_jobs, prefer="threads") as parallel:
    i_dataset = 0
    for dataset_name in dataset_configs:
        dataset_config = dataset_configs[dataset_name]
        X, y_true = dataset_config["data"]()
        psi = dataset_config["psi"]
        contamination = dataset_config["contamination"]
        X_ = ball_scale(X)

        for estimator_name in estimator_configs:
            estimator_config = estimator_configs[estimator_name]

            if (
                estimator_name == "iforest_sklearn" or estimator_name == "iforest_mass"
            ) and dataset_name == "single_blob":
                estimator = estimator_config(2, contamination, parallel)
            elif estimator_name == "iforest_path" and (
                dataset_name == "two_blobs" or dataset_name == "imba_blobs"
            ):
                estimator = estimator_config(2, contamination, parallel)
            elif estimator_name == "iforest_mass" and dataset_name == "two_blobs":
                estimator = estimator_config(4, contamination, parallel)
            elif estimator_name == "iforest_path" and dataset_name == "two_moons":
                estimator = estimator_config(16, contamination, parallel)
            elif estimator_name.startswith("iforest_"):
                estimator = estimator_config(32, contamination, parallel)
            else:
                estimator = estimator_config(psi, contamination, parallel)

            t0 = time.time()
            y_pred = estimator.fit_predict(
                X_ if estimator_name in ["iforest_path", "iforest_mass"] else X
            )
            t = time.time() - t0
            y_pred = y_pred.astype(np.int64)
            # y_score = estimator.decision_function(X)

            plt.subplot(len(dataset_configs), len(estimator_configs), plot_num)
            if i_dataset == 0:
                plt.title(estimator_name, size=18)

            # plot the levels lines and the points
            Z = estimator.predict(
                xxyy_ if estimator_name in ["iforest_path", "iforest_mass"] else xxyy
            )
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")

            colors = np.array(["#377eb8", "#ff7f00"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

            plt.xlim(-7, 7)
            plt.ylim(-7, 7)
            plt.xticks(())
            plt.yticks(())
            plt.text(
                0.99,
                0.01,
                ("%.2fs" % (t)).lstrip("0"),
                transform=plt.gca().transAxes,
                size=15,
                horizontalalignment="right",
            )
            plot_num += 1
        i_dataset += 1
    plt.show()
