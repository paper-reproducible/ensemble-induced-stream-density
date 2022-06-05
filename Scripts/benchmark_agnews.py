import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone

from Metric import fmeasure
from Clustering import DensityPeak, Grinch
from IsolationEstimators import (
    DEMassEstimator,
    MassEstimator,
    DataIndependentDensityEstimator,
    DataIndependentEstimator,
)
from ArtificialStream._txt import TXTStream
from ArtificialStream._gaussians import gaussian2x2

from joblib import Parallel, delayed


def cluster_demass(psi, parallel=None, t=2000):
    e = DEMassEstimator(psi, t, parallel=parallel, n_jobs=64)
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "Adaptive DEMass (psi={}, partitinoing=IForest)".format(psi)
    return dp


def cluster_mass(psi, parallel=None, t=2000):
    e = MassEstimator(psi, t, parallel=parallel, n_jobs=64)
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "Adaptive Mass (psi={}, partitinoing=IForest)".format(psi)
    return dp


def cluster_di_iforest(psi, parallel=None, t=2000):
    e = DataIndependentEstimator(
        psi, t, partitioning_type="iforest", parallel=parallel, n_jobs=64
    )
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "Data Independent IForest (psi={}, partitinoing=IForest)".format(psi)
    return dp


def cluster_di_anne(psi, parallel=None, t=2000):
    e = DataIndependentEstimator(
        psi, t, partitioning_type="anne", parallel=parallel, n_jobs=64
    )
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "Data Independent aNNE (psi={}, partitinoing=aNNE)".format(psi)
    return dp


def cluster_di_demass(psi, parallel=None, t=2000):
    e = DataIndependentDensityEstimator(psi, t, parallel=parallel, n_jobs=64)
    dp = DensityPeak(-1, e)
    dp.accept_k = True
    dp.label = "Data Independent DEMass (psi={}, partitinoing=IForest)".format(psi)
    return dp


from OtherEstimators import RACEDensityEstimator, L2LSH, AdaptiveKernelDensityEstimator


def cluster_race(psi, bw, t=200):
    e = RACEDensityEstimator(psi, L2LSH(t, 100, bw))
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


def cluster_grinch():
    e = Grinch(-1)
    e.accept_k = True
    e.label = "Grinch"
    return e


def construct_estimators(parallel=None):
    # psis = [20, 50, 100, 200, 400]
    # psis = [10, 20, 30, 40, 60, 90]
    dp = []
    
    psis = [i for i in range(26, 35)]
    dp += [cluster_di_demass(psi, parallel) for psi in psis]
    dp += [cluster_di_iforest(psi, parallel) for psi in psis]
    psis = [i for i in range(10, 14)]
    dp += [cluster_di_anne(psi, parallel) for psi in psis]
    
    psis = [i for i in range(4, 13)]
    dp += [cluster_mass(psi, parallel) for psi in psis]
    dp += [cluster_demass(psi, parallel) for psi in psis]

    bws = [0.1, 0.2, 0.3, 0.4, 0.5]
    psis = [2, 3, 4, 5, 6]
    for bw in bws:
        dp += [cluster_race(psi, bw) for psi in psis]
        dp += [cluster_kde(psi, bw, parallel) for psi in psis]

    # grinch = [cluster_grinch()]

    return dp # + grinch


def benchmark_stream(stream: TXTStream, estimator, evaluate_window=1000):
    # stream = ProbabilityStream()
    stream.reset()

    X = None
    y = None

    end = False

    f1_rt = []

    while not end:  # data is none when ts=0
        # for i in range(505):  # data is none when ts=0
        X_i, y_i, end = stream.next_tik(2000)

        # fit
        if X_i.shape[0] > 0:
            X = X_i if X is None else np.append(X, X_i, axis=0)
            y = np.array(y_i) if y is None else np.append(y, y_i, axis=0)

            # estimator = cluster_estimator(100)
            if estimator.fitted > 0:
                estimator.partial_fit(X_i)
            elif X.shape[0] >= evaluate_window:
                estimator.fit(X)

            f1 = 1.0
            if estimator.accept_k:
                k = np.unique(y).shape[0]
                if k > 1:
                    X_test = X[X.shape[0] - evaluate_window : X.shape[0], :]
                    y_true = y[y.shape[0] - evaluate_window : y.shape[0]]
                    y_pred = estimator.predict(X_test, k=k)
                    if y_pred.shape[0] != X_test.shape[0]:
                        y_pred = y_pred[y.shape[0] - evaluate_window : y.shape[0]]
                    f1, _, _ = fmeasure(y_true, y_pred)
            elif X.shape[0] >= evaluate_window:
                X_test = X[X.shape[0] - evaluate_window : X.shape[0], :]
                y_true = y[y.shape[0] - evaluate_window : y.shape[0]]
                y_pred = estimator.predict(X_test)
                f1, _, _ = fmeasure(y_true, y_pred)

            print("{}: {} => f1={}".format(X.shape[0], estimator.label, f1))

            f1_rt.append(f1)

    return X, y, f1_rt, estimator


def benchmark_total(X, y_true, estimator):

    try:
        check_is_fitted(estimator)
        if estimator.accept_k:
            y_pred = estimator.predict(X, k=np.unique(y_true).shape[0])
        else:
            y_pred = estimator.predict(X)
    except:
        if estimator.accept_k:
            y_pred = estimator.fit_predict(X, k=np.unique(y_true).shape[0])
        else:
            y_pred = estimator.fit_predict(X)
    f1, _, _ = fmeasure(y_true, y_pred)

    print("{} => f1(total)={}".format(estimator.label, f1))

    return f1


rt_estimators = construct_estimators()
static_estimators = construct_estimators()


def single_task(i):
    rt_estimator = rt_estimators[i]
    # stream = TXTStream("data/tsv/agnews.tsv")
    stream = TXTStream("data/tsv/agnews_manipulated.tsv")
    X, y, f1_rt, traind_estimator = benchmark_stream(stream, rt_estimator)

    static_estimator = static_estimators[i]
    f1_static = benchmark_total(X, y, static_estimator)

    f1_dynamic = benchmark_total(X, y, traind_estimator)

    estimator_label = rt_estimator.label

    return (f1_rt, f1_static, f1_dynamic, estimator_label)


def benchmark_agnews():

    # all_results = Parallel(n_jobs=8, backend="multiprocessing")(  # multiprocessing threading
    #     delayed(single_task)(i) for i in range(len(rt_estimators))
    # )

    all_results = []
    for i in range(len(rt_estimators)):
        all_results.append(single_task(i))

    f1_rt = [r[0] for r in all_results]
    f1_static = [r[1] for r in all_results]
    f1_dynamic = [r[2] for r in all_results]
    labels = [r[3] for r in all_results]

    with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
        "benchmark_agnews.xlsx"
    ) as writer:

        # df_X = pd.DataFrame(X)
        # df_X.to_excel(writer, sheet_name="X")

        df_f1_rt = pd.DataFrame(f1_rt).transpose()
        df_f1_rt.columns = labels
        df_f1_rt.to_excel(writer, "F1 Sliding 500")

        df_static = pd.DataFrame([f1_static])
        df_static.to_excel(writer, "F1 one-pass")

        df_dynamic = pd.DataFrame([f1_dynamic])
        df_dynamic.to_excel(writer, "F1 incremental")

if __name__ == "__main__":
    benchmark_agnews()