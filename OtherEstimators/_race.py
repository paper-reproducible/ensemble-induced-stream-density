from abc import ABCMeta
import numpy as np
from sklearn.base import BaseEstimator, DensityMixin
from Common import get_array_module


class RACEDensityEstimator(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self, hash_range, hasher):
        self.hasher = hasher
        self.t = hasher.reps
        self.S = RACE(hasher.reps, hash_range)
        self.psi = hash_range

    def fit(self, X, y=None):
        xp, _ = get_array_module(X)
        self.S.clear(xp)
        self.partial_fit(X)
        return self

    def partial_fit(self, X, y=None):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        for x in X:
            self.S.add(self.hasher.hash(x))
        return self

    def score(self, X):
        xp, _ = get_array_module(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        result = []
        for x in X:
            result.append(self.S.query(self.hasher.hash(x)))
        return xp.array(result)


class RACE:
    def __init__(self, repetitions, hash_range):
        self.R = repetitions  # number of ACEs (rows) in the array
        self.W = hash_range  # range of each ACE (width of each row)
        self.counts = None  # np.zeros((self.R, self.W), dtype=self.dtype)

    def rehash(self, hashvalues):
        xp, xpUtils = get_array_module(hashvalues)
        if self.counts is None:
            self.counts = xp.zeros((self.R, self.W), dtype=np.dtype(int))

        rehashed = xp.floor(hashvalues)
        rehashed = xpUtils.cast(rehashed % self.W, dtype=np.dtype(int))
        return rehashed, xp

    def add(self, hashvalues):
        rehashed, xp = self.rehash(hashvalues)
        self.counts[xp.arange(self.counts.shape[0]), rehashed] += 1
        return

    def remove(self, hashvalues):
        for idx, hashvalue in enumerate(hashvalues):
            rehash = int(hashvalue)
            rehash = rehash % self.W
            self.counts[idx, rehash] += -1

    def clear(self, xp):
        self.counts = xp.zeros((self.R, self.W), dtype=np.dtype(int))

    def query(self, hashvalues):
        rehashed, xp = self.rehash(hashvalues)
        return xp.average(self.counts[xp.arange(self.counts.shape[0]), rehashed])


class L2LSH:
    def __init__(self, N, d, r):
        # N = number of hashes
        # d = dimensionality
        # r = "bandwidth" or "sigma"
        self.reps = N
        self.N = N
        self.d = d
        self.r = r
        self.bw = r

        # set up the gaussian random projection vectors
        self.W = None  # np.random.normal(size=(N, d))
        self.b = None  # np.random.uniform(low=0, high=r, size=N)

    def hash(self, x):
        xp, _ = get_array_module(x)
        if self.W is None:
            self.W = xp.random.standard_normal(size=(self.N, self.d))
        if self.b is None:
            self.b = xp.random.uniform(low=0, high=self.r, size=[self.N])
        return (xp.squeeze(xp.dot(self.W, x)) + self.b) / self.r


class FastSRPMulti:
    # multiple SRP hashes combined into a set of N hash codes
    def __init__(self, reps, d, p):
        # reps = number of hashes (reps)
        # d = dimensionality
        # p = "bandwidth" = number of hashes (projections) per hash code
        self.reps = reps
        self.N = int(reps * p)  # number of hash computations
        self.N_codes = int(reps)  # number of output codes
        self.d = int(d)
        self.p = int(p)

        # set up the gaussian random projection vectors
        self.W = None  # np.random.normal(size=(self.N, d))
        self.powersOfTwo = None  # np.array([2 ** i for i in range(self.p)])

    def hash(self, x):
        xp, _ = get_array_module(x)

        if self.W is None:
            self.W = xp.random.normal(size=(self.N, self.d))
        if self.powersOfTwo is None:
            self.powersOfTwo = xp.array([2**i for i in range(self.p)])
        # p is the number of concatenated hashes that go into each
        # of the final output hashes
        h = xp.sign(xp.dot(self.W, x))
        h = xp.clip(h, 0, 1)
        h = xp.reshape(h, (self.N_codes, self.p))
        return xp.dot(h, self.powersOfTwo)


# import matplotlib.pyplot as plt


# sigma = 10.0  # bandwidth of KDE
# dimensions = 2

# # RACE sketch parameters
# reps = 80
# hash_range = 20

# lsh = L2LSH(reps, dimensions, sigma)
# S = RACEDensity(hash_range, hasher=lsh)

# # Generate data
# np.random.seed(420)

# N = 500
# cov1 = np.array([[60, 0], [0, 60]])
# cov2 = np.array([[40, 0], [0, 40]])
# cov3 = np.array([[40, -30], [-30, 40]])
# cov4 = np.array([[50, 40], [40, 60]])
# D1a = np.random.multivariate_normal([-30, -30], cov1, N)
# D1b = np.random.multivariate_normal([30, 30], cov2, N)
# D1c = np.random.multivariate_normal([-10, 10], cov3, N)
# D1d = np.random.multivariate_normal([20, -15], cov4, N)
# data = np.vstack((D1a, D1b, D1c, D1d))

# # feed data into RACE
# S.fit(data)

# xmin, xmax = (-80, 80)
# ymin, ymax = (-80, 80)

# M = 200
# x = np.linspace(xmin, xmax, M)
# y = np.linspace(ymin, ymax, M)
# Z = np.zeros((M, M))

# for i, xi in enumerate(x):
#     for j, yi in enumerate(y):
#         Z[j, i] = S.predict(np.array([xi, yi]))

# X, Y = np.meshgrid(x, y)

# cs = plt.contourf(
#     X, Y, Z, 15, cmap=plt.cm.get_cmap("RdBu_r"), vmax=np.max(Z), vmin=np.min(Z)
# )
# plt.plot(data[:, 0], data[:, 1], "k.", fillstyle="none", markersize=4, alpha=0.2)
# plt.show()
