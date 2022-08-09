import numpy as np
import pandas as pd
import h5py
import scipy


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


# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py
def load_sklearn_artificial(name, xp):
    from sklearn.datasets import make_moons, make_blobs

    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)

    X, y = (None, None)

    if name == "two_moons":
        X, y = make_moons(n_samples=(n_outliers, n_inliers), noise=0.05, random_state=0)
        X = xp.array(4.0 * (X - np.array([0.5, 0.25])))
        y = xp.where(y == 0, -1, y)
    elif name == "uniform":
        X = 14.0 * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)
        X = xp.array(X)
        y = xp.ones(n_samples)
    else:
        rng = np.random.RandomState(42)
        if name == "one_blob":
            X = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, **blobs_params)[0]
        if name == "two_blobs":
            X = make_blobs(
                centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5], **blobs_params
            )[0]
        if name == "imbalanced":
            X = make_blobs(
                centers=[[2, 2], [-2, -2]], cluster_std=[1.5, 0.3], **blobs_params
            )[0]
        X = xp.concatenate(
            [X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0
        )
        y = xp.concatenate([np.ones(n_inliers), -np.ones(n_outliers)], axis=0)

    return X, y


# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_outlier_detection_bench.html
def load_sklearn_real(dataset_name, xp=np):
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

    X = xp.array(X, dtype=float)
    y = xp.array(y, dtype=int)
    y = xp.where(y == 1, -1, y)
    y = xp.where(y == 0, 1, y)
    return X, y
