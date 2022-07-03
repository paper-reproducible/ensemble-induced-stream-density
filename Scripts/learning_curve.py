from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error as mse
from ArtificialStream._mixture import Mixture, minMaxNormalise
from Common import call_by_argv, ball_scale

_parallel = Parallel(n_jobs=1, prefer="threads")


def gmm_dens(X, X_sub, k):
    gm = GaussianMixture(n_components=k).fit(X_sub)
    prob = np.exp(gm.score_samples(X))
    return prob


def egmm_dens(X, k, psi=5000, t=1000, paralell=_parallel):
    choices = np.random.choice(psi, size=(t, X.shape[0]))
    probs = paralell(delayed(gmm_dens)(X, X[choices[i, :], :], k) for i in range(t))
    return np.average(probs, axis=0)


def demass(X, psi, t=1000, parallel=_parallel):
    from IsolationEstimators import DEMassEstimator

    X = ball_scale(X)
    e = DEMassEstimator(psi, t, parallel=parallel)
    e.fit(X)
    probs = e.score(X)
    return probs


def gaussian_mixture(n_components, dims):
    mix = Mixture()
    l = np.random.rand(n_components, dims + 2)
    for i in range(n_components):
        loc = l[i, :dims]
        scale = l[i, dims] / n_components
        weight = l[i, dims + 1]
        mix.add(weight, "gaussian", loc=loc, scale=scale)

    return mix


def normalised_mse(y_true, y_pred):
    y_true = minMaxNormalise(y_true)
    y_pred = minMaxNormalise(y_pred)
    return mse(y_true, y_pred)


current_folder = "./"
excel_filename = "egmm_learning_curve.xlsx"


def learning_curve(
    folder=current_folder, n=10000, dims=2, n_gaussians=4, debug=False, show_data=False
):
    import os.path

    n = int(n)
    dims = int(dims)
    n_gaussians = int(n_gaussians)

    if not folder.endswith("/"):
        folder += "/"
    if not os.path.isdir(folder):
        raise Exception("Destination folder (" + folder + ") does not exist")

    mix = gaussian_mixture(n_gaussians, dims)
    X = mix.sample(n, dims)
    p_true = mix.prob(X)

    if show_data:
        import matplotlib.pyplot as plt
        from ArtificialStream._plot import scatter

        X_ = minMaxNormalise(X)
        fig = plt.figure()
        fig.suptitle("Normalised data")
        scatter(X_[:, 0], X_[:, 1], fig=fig)

    with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
        folder + excel_filename
    ) as writer:

        df_X = pd.DataFrame(X)
        df_X.columns = ["dim_" + str(i) for i in range(dims)]
        df_X.to_excel(writer, sheet_name="X")

        with (
            Parallel(n_jobs=1, prefer="threads")
            if debug
            else Parallel(n_jobs=16, prefer="processes")
        ) as p:
            # EGMM
            l_k = [4, 8, 16] if debug else [4, 8, 16, 32, 64]
            l_psi = [500, 1000, 2000] if debug else [500, 1000, 2000, 4000, 8000]
            l_t = [100, 200] if debug else [100, 200, 400]

            ll_pred = []
            ll_pred.append([-1, -1, -1] + p_true.tolist())
            ll_errors = []
            for k in l_k:
                for psi in l_psi:
                    for t in l_t:
                        print("EGMM", k, psi, t)

                        p_pred = egmm_dens(X, k, psi, t, p)
                        ll_pred.append([k, psi, t] + p_pred.tolist())

                        mse = normalised_mse(p_true, p_pred)
                        ll_errors.append([k, psi, t, mse])

            df_pred = pd.DataFrame(ll_pred)
            df_pred.columns = ["k", "psi", "t"] + [str(i) for i in range(n)]
            df_pred.to_excel(writer, sheet_name="Density (EGMM)")

            df_error = pd.DataFrame(ll_errors)
            df_error.columns = ["k", "psi", "t", "mse"]
            df_error.to_excel(writer, sheet_name="Errors (EGMM)")

            # DEMass
            l_psi = (
                [4, 8, 16] if debug else [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            )
            l_t = [100, 200] if debug else [100, 200, 400]

            ll_pred = []
            ll_pred.append([-1, -1] + p_true.tolist())
            ll_errors = []
            for psi in l_psi:
                for t in l_t:
                    print("DEMass", psi, t)

                    p_pred = demass(X, psi, t, p)
                    ll_pred.append([psi, t] + p_pred.tolist())

                    mse = normalised_mse(p_true, p_pred)
                    ll_errors.append([psi, t, mse])

            df_pred = pd.DataFrame(ll_pred)
            df_pred.columns = ["psi", "t"] + [str(i) for i in range(n)]
            df_pred.to_excel(writer, sheet_name="Density (DEMass)")

            df_error = pd.DataFrame(ll_errors)
            df_error.columns = ["psi", "t", "mse"]
            df_error.to_excel(writer, sheet_name="Errors (DEMass)")

    return


if __name__ == "__main__":
    # PYTHONPATH=. python Scripts/learning_curve.py ./ debug=True show_data=True
    import sys

    sys.argv += ["debug=True", "show_data=True"]
    call_by_argv(learning_curve)
    print("done")
