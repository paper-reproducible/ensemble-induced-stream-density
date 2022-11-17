import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

from sklearn.metrics import roc_auc_score
from Data._outliers import load_sklearn_artificial
from Common import ball_scale


dataset_configs = {
    "demo": {
        "data": lambda xp=np: (
            xp.array(
                [[-5.1, -5.1], [-4.6, -4.6], [-1.1, -1.1], [-0.6, -0.8], [6.6, 6.6]]
            ),
            xp.array([1, 1, 1, 1, -1]),
        ),
        "contamination": 0.2,
        "psi": 2,
    },
    "single_blob": {
        "data": lambda xp=np: load_sklearn_artificial("one_blob", xp),
        "contamination": 0.15,
        "psi": 2,
    },
    "two_blobs": {
        "data": lambda xp=np: load_sklearn_artificial("two_blobs", xp),
        "contamination": 0.15,
        "psi": 8,  # 32,
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

_t = 100
_n_jobs = 32

estimator_configs = {
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
        rotation=True,
        # rotation=False,
        # global_boundaries=(np.array([-7.0, -7.0]), np.array([7.0, 7.0])),
    ),
    "iforest_mass": lambda psi, contamination, parallel: IsolationForestAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=True,
        parallel=parallel,
        rotation=True,
        # rotation=False,
        # global_boundaries=(np.array([-7.0, -7.0]), np.array([7.0, 7.0])),
    ),
    "inne_ratio": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=False,
        isolation_model="inne",
        parallel=parallel,
    ),
    "inne_mass": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=True,
        isolation_model="inne",
        parallel=parallel,
    ),
    "anne_dis": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=False,
        isolation_model="anne",
        parallel=parallel,
    ),
    "anne_mass": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=True,
        isolation_model="anne",
        parallel=parallel,
    ),
    "soft_anne_dis": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=False,
        isolation_model="soft_anne",
        parallel=parallel,
    ),
    "soft_anne_mass": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=True,
        isolation_model="soft_anne",
        parallel=parallel,
    ),
    "fuzzy_mass": lambda psi, contamination, parallel: IsolationBasedAnomalyDetector(
        psi,
        t=_t,
        contamination=contamination,
        mass_based=True,
        isolation_model="fuzzy",
        parallel=parallel,
    ),
}

titles = {
    "fuzzy_mass": "Fuzzy Mass",
    "inne_mass": "iNNE Mass",
    "inne_ratio": "iNNE Score",
    "anne_mass": "aNNE Mass",
    "anne_dis": "aNNE Score",
    "soft_anne_mass": "Soft aNNE Mass",
    "soft_anne_dis": "Soft aNNE Score",
    "iforest_sklearn": "iForest Score",
    "iforest_path": "iForest Rotated Score",
    "iforest_mass": "iForest Rotated Mass",
}

matplotlib.rcParams["contour.negative_linestyle"] = "solid"


# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
xxyy = np.c_[xx.ravel(), yy.ravel()]

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

        scaled = ball_scale(np.concatenate([X, xxyy], axis=0))
        X_ = scaled[: X.shape[0], :]
        xxyy_ = scaled[X.shape[0] :, :]

        for estimator_name in estimator_configs:
            estimator_config = estimator_configs[estimator_name]
            estimator = None

            if estimator_name.startswith("iforest_"):
                if dataset_name == "single_blog":
                    if estimator_name == "iforest_mass":
                        estimator = estimator_config(2, contamination, parallel)
                    elif estimator_name == "iforest_path":
                        estimator = estimator_config(32, contamination, parallel)
                elif dataset_name == "two_blobs":
                    if estimator_name == "iforest_mass":
                        estimator = estimator_config(4, contamination, parallel)
                    elif estimator_name == "iforest_path":
                        estimator = estimator_config(32, contamination, parallel)
                elif dataset_name == "imba_blobs":
                    if estimator_name == "iforest_mass":
                        estimator = estimator_config(2, contamination, parallel)
                    elif estimator_name == "iforest_path":
                        estimator = estimator_config(32, contamination, parallel)
                elif dataset_name == "two_moons":
                    if estimator_name == "iforest_mass":
                        estimator = estimator_config(2, contamination, parallel)
                    elif estimator_name == "iforest_path":
                        estimator = estimator_config(32, contamination, parallel)
                elif dataset_name == "demo":
                    estimator = estimator_config(4, contamination, parallel)
                else:
                    estimator = estimator_config(32, contamination, parallel)

            if estimator is None:
                estimator = estimator_config(psi, contamination, parallel)

            t0 = time.time()
            y_pred = estimator.fit_predict(
                X_
                if estimator_name in ["iforest_path", "iforest_mass"]
                else X
                # X
            )
            t = time.time() - t0
            y_pred = y_pred.astype(np.int64)
            y_score = estimator.decision_function(
                X_
                if estimator_name in ["iforest_path", "iforest_mass"]
                else X
                # X
            )
            roc_auc = roc_auc_score(y_true, y_score)

            ax = plt.subplot(len(dataset_configs), len(estimator_configs), plot_num)
            if i_dataset == 0:
                plt.title(titles[estimator_name], size=18)

            # plot the levels lines and the points
            # Z = estimator.predict(
            #     xxyy_ if estimator_name in ["iforest_path", "iforest_mass"] else xxyy
            # )
            # Z = Z.reshape(xx.shape)
            # plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")

            if estimator_name == "fuzzy_mass":
                Z = np.log(estimator.score_samples(xxyy))
            else:
                Z = estimator.decision_function(
                    xxyy_
                    if estimator_name in ["iforest_path", "iforest_mass"]
                    else xxyy
                    # xxyy
                )
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z)

            # Z = estimator.decision_function(
            #     xxyy_ if estimator_name in ["iforest_path", "iforest_mass"] else xxyy
            # )
            # Z = Z.reshape(xx.shape)
            # plt.contour(xx, yy, Z, 3, colors='black')

            colors = np.array(["#377eb8", "#ff7f00"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])
            # plt.scatter(X[:, 0], X[:, 1], s=10, color="red")

            # if hasattr(estimator, "iforest"):
            #     r = 0
            #     for i in estimator.iforest.transformers_:
            #         s = i.samples_
            #         plt.scatter(s[:, 0], s[:, 1], s=10)
            #         boundaries = i.combine_boundaries()
            #         for i_node in range(boundaries.shape[0]):
            #             lower = boundaries[i_node, 0, :]
            #             upper = boundaries[i_node, 1, :]
            #             rect = patches.Rectangle(
            #                 (lower[0], lower[1]),
            #                 upper[0] - lower[0],
            #                 upper[1] - lower[1],
            #                 linewidth=1,
            #                 edgecolor="r",
            #                 facecolor="none",
            #             )
            #             # Add the patch to the Axes
            #             ax.add_patch(rect)

            #         r = r + 1
            #         if r >= 4:
            #             break

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
