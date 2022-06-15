import math as m
import numpy as np


def _gen_normal(xp, n, dims, loc, scale):
    X = loc + scale * xp.random.standard_normal(size=[n, dims])
    return X


def _pdf_normal(xp, X, loc, scale):
    Z = (2 * m.pi * (scale**2)) ** 0.5
    p = xp.exp(-0.5 * xp.sum((X - loc) ** 2, axis=1) / (scale**2)) / Z
    return p


def _prepare_mix(xp, gen, pdf, **kwargs):
    reserved = ["n", "dims", "X"]
    settings = {}
    for key in kwargs:
        if key not in reserved:
            settings[key] = kwargs[key]

    def run_gen(n, dims):
        params = {
            "xp": xp,
            "n": n,
            "dims": dims,
            **settings,
        }

        return gen(**params)

    def run_pdf(X):
        params = {
            "xp": xp,
            "X": X,
            **settings,
        }
        return pdf(**params)

    return settings, run_gen, run_pdf


def minMaxNormalise(X):
    X_ = X - np.min(X, axis=0, keepdims=True)
    X_ = X_ / np.max(X_, axis=0, keepdims=True)
    return X_


class Mixture:
    def __init__(self, xp=np):
        self.generators = []
        self.pdfs = []
        self.weights = []
        self.settings = []
        self.xp = xp

    def add(self, weight, name, **kwargs):
        xp = self.xp
        self.weights.append(weight)
        if name == "normal" or name == "gaussian":
            params, gen, pdf = _prepare_mix(xp, _gen_normal, _pdf_normal, **kwargs)
        else:
            raise ValueError("wrong name")
        self.pdfs.append(pdf)
        self.generators.append(gen)
        self.settings.append(params)
        return

    def sample(self, n, dims):
        xp = self.xp
        result = xp.zeros([n, dims])
        n_components = len(self.weights)
        if "choice" in dir(xp.random):
            p = xp.array(self.weights) / xp.sum(self.weights)
            choice = xp.random.choice(n_components, n, p=p)
        else:
            p = np.array(self.weights) / np.sum(self.weights)
            choice = np.random.choice(n_components, n, p=p)
        for i in range(n_components):
            gen = self.generators[i]
            result = xp.where(xp.expand_dims(choice, axis=1) == i, gen(n, dims), result)

        return result

    def prob(self, X):
        xp = self.xp
        n_components = len(self.weights)
        p = xp.array(self.weights) / xp.sum(self.weights)
        result = xp.zeros([X.shape[0]])
        for i in range(n_components):
            pdf = self.pdfs[i]
            result = result + pdf(X) * p[i]

        return result


if __name__ == "__main__":
    from _plot import scatter

    # mix = Mixture()

    import tensorflow as tf

    tnp = tf.experimental.numpy
    mix = Mixture(xp=tnp)

    mix.add(2, "gaussian", loc=[0.3, 0.4], scale=0.02)
    mix.add(1.5, "gaussian", loc=[0.1, 0.4], scale=0.07)
    mix.add(1, "gaussian", loc=[0.5, 0.5], scale=0.01)

    X = mix.sample(10000, 2)
    print(type(X))
    print(mix.prob(X).shape)
    X_ = minMaxNormalise(X)
    scatter(X_[:, 0], X_[:, 1])
