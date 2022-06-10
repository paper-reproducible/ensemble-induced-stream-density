import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted

from Common import get_array_module_with_utils
from Metric import fmeasure
from Clustering import DensityPeak
from IsolationEstimators import (
    DEMassEstimator,
    MassEstimator,
    DataIndependentDensityEstimator,
    DataIndependentEstimator,
)
from ArtificialStream._stream import ProbabilityStream
from ArtificialStream._gaussians import gaussian2x2_ball

from joblib import Parallel


def cluster_demass(psi, parallel=None, t=200):
    e = DEMassEstimator(psi, t, parallel=parallel, n_jobs=64)
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "Adaptive DEMass (psi={}, partitinoing=IForest)".format(psi)
    return dp


def cluster_mass(psi, parallel=None, t=200):
    e = MassEstimator(psi, t, parallel=parallel, n_jobs=64)
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "Adaptive Mass (psi={}, partitinoing=IForest)".format(psi)
    return dp


def cluster_di_iforest(psi, parallel=None, t=200):
    e = DataIndependentEstimator(
        psi, t, partitioning_type="iforest", parallel=parallel, n_jobs=64
    )
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "Data Independent IForest (psi={}, partitinoing=IForest)".format(psi)
    return dp


def cluster_di_anne(psi, parallel=None, t=200):
    e = DataIndependentEstimator(
        psi, t, partitioning_type="anne", parallel=parallel, n_jobs=64
    )
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "Data Independent aNNE (psi={}, partitinoing=aNNE)".format(psi)
    return dp


def cluster_di_demass(psi, parallel=None, t=200):
    e = DataIndependentDensityEstimator(psi, t, parallel=parallel, n_jobs=64)
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "Data Independent DEMass (psi={}, partitinoing=IForest)".format(psi)
    return dp


from OtherEstimators import RACEDensityEstimator, L2LSH, AdaptiveKernelDensityEstimator


def cluster_race(psi, bw, t=200):
    e = RACEDensityEstimator(psi, L2LSH(t, 2, bw))
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "RACE (hash_range={}, bandwidth={}, lsh=L2)".format(psi, bw)
    return dp


def cluster_kde(psi, bw, parallel=None, t=200):
    e = AdaptiveKernelDensityEstimator(psi, t, bandwidth=bw, parallel=parallel)
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "KDE (hash_range={}, bandwidth={}, kernel=Gaussian)".format(psi, bw)
    return dp


def construct_estimators(parallel=None):
    # psis = [20, 50, 100, 200, 400]
    psis = [5]
    dp = []
    dp += [cluster_di_demass(psi, parallel) for psi in psis]
    dp += [cluster_di_iforest(psi, parallel) for psi in psis]
    dp += [cluster_di_anne(psi, parallel) for psi in psis]
    dp += [cluster_mass(psi, parallel) for psi in psis]
    dp += [cluster_demass(psi, parallel) for psi in psis]

    # bws = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    bws = [0.05]
    for bw in bws:
        # dp += [cluster_race(psi, bw) for psi in psis]
        dp += [cluster_kde(psi, bw, parallel) for psi in psis]

    return dp


def benchmark_stream(
    stream: ProbabilityStream, drifts, n, estimators, evaluate_window=500
):
    xp = stream.xp
    # stream = ProbabilityStream()
    stream.reset()

    X = None
    f1_rt = [None for _ in range(len(estimators))]

    for i in range(n + 1):  # data is none when ts=0
        # for i in range(505):  # data is none when ts=0
        X_i, _, _ = stream.next_tik()

        # fit
        if X_i is not None:
            X = X_i if X is None else xp.append(X, X_i, axis=0)

            def task(estimator):
                # estimator = cluster_estimator(100)
                if estimator.fitted > 0:
                    estimator.partial_fit(X_i)
                elif X.shape[0] >= evaluate_window:
                    estimator.fit(X)

                f1 = 1.0
                if estimator.accept_k:
                    k = sorted(drifts + [i]).index(i)
                    if k > 1:
                        X_test = X[X.shape[0] - evaluate_window : X.shape[0], :]
                        y_true = stream.current_truth(X_test)
                        y_pred = estimator.predict(X_test, k=k)
                        f1, _, _ = fmeasure(y_true, y_pred)
                elif X.shape[0] >= evaluate_window:
                    X_test = X[X.shape[0] - evaluate_window : X.shape[0], :]
                    y_true = stream.current_truth(X_test)
                    y_pred = estimator.predict(X_test)
                    f1, _, _ = fmeasure(y_true, y_pred)

                print("{}: {} => f1={}".format(i, estimator.label, f1))
                return f1

            for j in range(len(estimators)):
                estimator = estimators[j]
                # f1 = l_f1[j]
                f1 = task(estimator)
                f1_rt[j] = (
                    xp.array([f1])
                    if f1_rt[j] is None
                    else xp.append(f1_rt[j], xp.array([f1]))
                )

    return X, f1_rt, estimators


def benchmark_total(X, estimators, stream: ProbabilityStream):

    f1_static = [0 for _ in range(len(estimators))]

    for j in range(len(estimators)):
        estimator = estimators[j]
        # estimator = cluster_estimator(100)

        y_true = stream.current_truth(X)

        try:
            check_is_fitted(estimator)
            if estimator.accept_k:
                y_pred = estimator.predict(X, k=len(stream.pdf))
            else:
                y_pred = estimator.predict(X)
        except:
            if estimator.accept_k:
                y_pred = estimator.fit_predict(X, k=len(stream.pdf))
            else:
                y_pred = estimator.fit_predict(X)
        f1, _, _ = fmeasure(y_true, y_pred)
        f1_static[j] = f1
        print("{} => f1(total)={}".format(estimator.label, f1))

    return f1_static


def benchmark_gaussian2x2():
    # with Parallel(n_jobs=32, backend="loky") as parallel:
    # with Parallel(n_jobs=32, backend="multiprocessing") as parallel:
    # with Parallel(n_jobs=32, backend="threading") as parallel:
    with Parallel(n_jobs=1, prefer="threads") as parallel:

        stream = None

        if stream is None:
            try:
                xp, xpUtils = get_array_module_with_utils("tf.numpy")
                stream = ProbabilityStream(ball=True, xp=xp, linalg=xpUtils)
            except:
                pass

        if stream is None:
            try:
                xp, xpUtils = get_array_module_with_utils("cupy")
                stream = ProbabilityStream(ball=True, xp=xp, linalg=xpUtils)
            except:
                pass

        if stream is None:
            xp, xpUtils = get_array_module_with_utils("numpy")
            stream = ProbabilityStream(ball=True, xp=xp, linalg=xpUtils)

        name, drifts, n = gaussian2x2_ball(stream)

        # keep this out without writer
        X, f1_rt, trained_estimators = benchmark_stream(
            stream, drifts, n, construct_estimators(parallel)
        )

        more_estimators = construct_estimators(parallel)
        f1_static = benchmark_total(X, more_estimators, stream)

        f1_dynamic = benchmark_total(X, trained_estimators, stream)

    with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
        "benchmark_gaussian2x2_ball.xlsx"
    ) as writer:

        df_X = pd.DataFrame(X)
        df_X.to_excel(writer, sheet_name="X")

        df_f1_rt = pd.DataFrame(f1_rt).transpose()
        df_f1_rt.columns = [e.label for e in trained_estimators]
        df_f1_rt.to_excel(writer, "F1 Sliding 500")

        df_static = pd.DataFrame([f1_static])
        df_static.to_excel(writer, "F1 one-pass")

        df_dynamic = pd.DataFrame([f1_dynamic])
        df_dynamic.to_excel(writer, "F1 incremental")


if __name__ == "__main__":
    benchmark_gaussian2x2()
