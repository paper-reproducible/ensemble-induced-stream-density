from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error as mse
from ArtificialStream._mixture import Mixture, minMaxNormalise

_parallel = Parallel(n_jobs=1, prefer="threads")


def gmm_dens(X, X_sub, k):
    gm = GaussianMixture(n_components=k).fit(X_sub)
    prob = np.exp(gm.score_samples(X))
    return prob


def egmm_dens(X, k, psi=5000, t=1000, paralell=_parallel):
    choices = np.random.choice(psi, size=(t, X.shape[0]))
    probs = paralell(delayed(gmm_dens)(X, X[choices[i, :], :], k) for i in range(t))
    return np.average(probs, axis=0)


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


if __name__ == "__main__":
    from ArtificialStream._plot import scatter, plot
    import matplotlib.pyplot as plt

    dims = 2

    # if dims == 1:
    #     n_guassians = 4
    #     mix = gaussian_mixture(n_guassians, dims)
    #     n = 10000
    #     X = mix.sample(n, dims)
    #     X = np.sort(X)
    #     p_true = mix.prob(X)

    #     X_ = minMaxNormalise(X)
    #     scatter(X_[:, 0], minMaxNormalise(p_true))

    #     # l_k = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    #     l_k = [8, 16]
    #     # l_psi = [1000, 2000, 4000, 8000]
    #     l_psi = [2000]
    #     # l_t = [100, 200, 400, 800, 1600]
    #     l_t = [100, 200]
    #     m_pred = np.zeros([len(l_k), len(l_psi), len(l_t), n])
    #     with Parallel(n_jobs=16, prefer="threads") as p:
    #         for i_k in range(len(l_k)):
    #             k = l_k[i_k]
    #             for i_psi in range(len(l_psi)):
    #                 psi = l_psi[i_psi]
    #                 for i_t in range(len(l_t)):
    #                     t = l_t[i_t]
    #                     p_pred = egmm_dens(X, k, psi, t, p)
    #                     m_pred[i_k, i_psi, i_t, :] = p_pred

    #                     scatter(X_[:, 0], minMaxNormalise(p_pred))
    #                     print((k, psi, t))

    if dims == 2:
        n_guassians = 4
        mix = gaussian_mixture(n_guassians, dims)
        n = 10000
        X = mix.sample(n, dims)
        X = np.sort(X)
        p_true = mix.prob(X)

        X_ = minMaxNormalise(X)

        l_k = [4, 8, 16, 32, 64]
        # l_k = [4, 8, 16]
        l_psi = [500, 1000, 2000, 4000, 8000]
        # l_psi = [5000]
        # l_t = [100, 200, 400, 800, 1600]
        l_t = [100, 200, 300, 400, 500]
        m_pred = np.zeros([len(l_k), len(l_psi), len(l_t), n])
        m_errors = np.zeros([len(l_k), len(l_psi), len(l_t)])
        # with Parallel(n_jobs=16, prefer="threads") as p:
        with Parallel(n_jobs=16, prefer="processes") as p:
            for i_k in range(len(l_k)):
                k = l_k[i_k]
                for i_psi in range(len(l_psi)):
                    psi = l_psi[i_psi]
                    for i_t in range(len(l_t)):
                        t = l_t[i_t]
                        print(k, psi, t)
                        p_pred = egmm_dens(X, k, psi, t, p)
                        m_pred[i_k, i_psi, i_t, :] = p_pred
                        m_errors[i_k, i_psi, i_t] = normalised_mse(p_true, p_pred)

        for i_t in range(len(l_t)):
            plot(l_k, m_errors[:, 0, i_t], show=False)

        with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
            "egmm_learning_curve.xlsx"
        ) as writer:

            df_X = pd.DataFrame(X)
            df_X.to_excel(writer, sheet_name="X")

            l_errors = []

            for i_k in range(len(l_k)):
                k = l_k[i_k]
                for i_psi in range(len(l_psi)):
                    psi = l_psi[i_psi]
                    for i_t in range(len(l_t)):
                        t = l_t[i_t]
                        e = m_errors[i_k, i_psi, i_t]
                        l_errors.append([k, psi, t, e])

            df_dynamic = pd.DataFrame(l_errors)
            df_dynamic.to_excel(writer, "Errors (k, psi, t, e)")

    plt.show()
    print("done")
