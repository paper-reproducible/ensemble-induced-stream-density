import os
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
from joblib import Parallel
from sklearn.metrics import (
    mutual_info_score as MI,
    adjusted_mutual_info_score as AMI,
    normalized_mutual_info_score as NMI,
    rand_score as RI,
    adjusted_rand_score as ARI,
)
from Common import (
    get_array_module,
    ball_scale,
    save_csv,
    save_parquet,
    init_xp,
)
from Clustering import DBSCAN
from Clustering._dbscan import epsilon_neighbourhood_density
from Data._clusters import load_mat


def metric(y_true, y_pred):
    mi = MI(y_true, y_pred)
    ami = AMI(y_true, y_pred)
    nmi = NMI(y_true, y_pred)
    ri = RI(y_true, y_pred)
    ari = ARI(y_true, y_pred)
    return mi, ami, nmi, ri, ari


def fitted_dbscan(name, psi, X, rotation=False, parallel=None):

    splits = name.split("_")
    model_name = "_".join(splits[:-1])
    suffix = splits[len(splits) - 1]

    tmp_eps = 1
    tmp_minPts = 1

    rotation = False  # Tired to tune rotated iforest

    X_ = (
        ball_scale(X)
        if rotation and model_name is not None and model_name.startswith("iforest")
        else X
    )

    if suffix == "similarity":
        m = DBSCAN(
            tmp_eps,
            tmp_minPts,
            psi,
            isolation_model_name=None if model_name == "l2" else model_name,
            use_alpha_neighbourhood_mass=True,
            rotation=rotation,
            parallel=parallel,
        )
        return m.fit(X_)

    if suffix == "anomaly":
        m = DBSCAN(
            tmp_eps,
            tmp_minPts,
            psi,
            isolation_model_name=model_name,
            use_alpha_neighbourhood_mass=False,
            use_anomaly_score=True,
            rotation=rotation,
            parallel=parallel,
        )
        return m.fit(X_)

    if suffix == "mass":
        m = DBSCAN(
            tmp_eps,
            tmp_minPts,
            psi,
            isolation_model_name=model_name,
            use_alpha_neighbourhood_mass=False,
            use_anomaly_score=False,
            rotation=rotation,
            parallel=parallel,
        )
        return m.fit(X_)


estimator_configs = {
    "l2_similarity": {"psi_values": [None]},
    "anne_similarity": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "soft_anne_similarity": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "inne_similarity": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "isotropic_similarity": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "iforest_similarity": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "anne_anomaly": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "soft_anne_anomaly": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "inne_anomaly": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "iforest_anomaly": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "anne_mass": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "soft_anne_mass": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "inne_mass": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "isotropic_mass": {"psi_values": [2, 4, 8, 16, 32, 64]},
    "iforest_mass": {"psi_values": [2, 4, 8, 16, 32, 64]},
}

data_dir = "Data/mat/clusters"
datasets = [
    "control",
    "dermatology",
    "diabetesC",
    "ecoli",
    "glass",
    "hard",
    "hill",
    "ILPD",
    "Ionosphere",
    "iris",
    "isolet",
    "libras",
    "liver",
    "LSVT",
    "musk",
    "Parkinsons",
    # "Pendig",
    "pima",
    "s1",
    "s2",
    "seeds",
    "Segment",
    "shape",
    "Sonar",
    # "spam",
    "SPECTF",
    "thyroid",
    "user",
    "vowel",
    "WDBC",
    "wine",
]
n_eps = 50
n_minPts = 50
n_rounds = 10


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    xp = np

    # import os
    # import tensorflow as tf

    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
    # tnp = tf.experimental.numpy
    # tnp.experimental_enable_numpy_behavior()
    # xp = tnp

    # import cupy as cp
    # xp=cp

    time_format = "%Y%m%d%H"
    save_folder = "Data/" + datetime.now().strftime(time_format) + "_dbscan"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with Parallel(n_jobs=32, prefer="threads") as parallel:
        for dataset_name in datasets:
            X, y = load_mat(data_dir, dataset_name, xp)
            _, xpUtils = get_array_module(X)
            y_true = xpUtils.to_numpy(y)
            n, dims = X.shape

            dataset_results = []

            for estimator_name in estimator_configs:
                estimator_config = estimator_configs[estimator_name]
                psi_values = estimator_config["psi_values"]

                for psi in psi_values:
                    for r in range(n_rounds if psi is not None else 1):
                        m = fitted_dbscan(
                            estimator_name, psi, X, rotation=False, parallel=parallel
                        )

                        sim_values = xp.sort(xpUtils.unique(m.m_sim.reshape([n * n])))[
                            1:
                        ]

                        eps_values = xp.linspace(
                            (sim_values[0] + sim_values[1]) / 2,
                            (
                                sim_values[sim_values.shape[0] - 2]
                                + sim_values[sim_values.shape[0] - 1]
                            )
                            / 2,
                            n_eps,
                        )

                        if m.l_dens is not None:
                            dens_values = xp.sort(xpUtils.unique(m.l_dens))[1:]
                            if dens_values.shape[0] < 2:
                                continue

                        for i in range(n_eps):
                            eps = xpUtils.to_numpy(eps_values[i])
                            m.eps = eps

                            l_dens = m.l_dens
                            if l_dens is None:
                                l_dens = epsilon_neighbourhood_density(
                                    m.m_sim, eps, m.using_similarity
                                )

                            dens_values = xp.sort(xpUtils.unique(l_dens))[1:]

                            if dens_values.shape[0] < 2:
                                continue

                            threshold_values = xp.linspace(
                                (dens_values[0] + dens_values[1]) / 2,
                                (
                                    dens_values[dens_values.shape[0] - 2]
                                    + dens_values[dens_values.shape[0] - 1]
                                )
                                / 2,
                                n_minPts,
                            )

                            for j in range(n_minPts):
                                minPts = xpUtils.to_numpy(threshold_values[j])
                                m.core_threshold = minPts
                                y_pred = m.predict(X=None)

                                y_pred = xpUtils.to_numpy(y_pred)

                                mi, ami, nmi, ri, ari = metric(y_true, y_pred)

                                print(
                                    dataset_name,
                                    estimator_name,
                                    np.array([psi]),
                                    np.array([r]),
                                    np.array([eps]),
                                    np.array([minPts]),
                                    np.array(
                                        [
                                            mi,
                                            ami,
                                            nmi,
                                            ri,
                                            ari,
                                        ]
                                    ),
                                )

                                dataset_results.append(
                                    [
                                        dataset_name,
                                        estimator_name,
                                        psi,
                                        r,
                                        eps,
                                        minPts,
                                        mi,
                                        ami,
                                        nmi,
                                        ri,
                                        ari,
                                    ]
                                )

            df_dataset_results = pd.DataFrame(dataset_results)
            df_dataset_results.columns = [
                "dataset",
                "isolation_method",
                "psi",
                "round",
                "eps",
                "minPts",
                "mi",
                "ami",
                "nmi",
                "ri",
                "ari",
            ]
            save_csv(
                df_dataset_results,
                save_folder
                + "/dbscan_"
                + dataset_name
                + "_"
                + datetime.now().strftime("%Y%m%d%H%M%S"),
            )
            # print(df_dataset_results)
