# kernel fuzzy 'partitioning'
from logging import root
import numpy as np
from sklearn.base import TransformerMixin
from Common import ReservoirSamplingEstimator, get_array_module
from Common._sampling import get_samples
from ArtificialStream._mixture import gaussian_mixture, minMaxNormalise


def _tf_train(
    X,
    psi,
    optimizer=None,
    n_epochs=200,
    print_epochs=False,
):
    import tensorflow as tf

    if optimizer is None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)
    import tensorflow_probability.python as tfp

    tfd = tfp.distributions
    tnp = tf.experimental.numpy

    _, dims = X.shape
    xp, xpUtils = get_array_module(X)

    if xp != tnp:
        raise Exception("We can only use Tensorflow optimisers for now.")

    const_locs = get_samples(X, psi)  # [n, psi]
    const_scale_base = xp.expand_dims(
        (
            xp.sort(
                xpUtils.pdist(const_locs, const_locs, root=False),
                axis=1,
            )[:, 1]
            / dims
        )
        ** 0.5,
        1,
    )  # [psi, 1]
    const_probs = xp.ones(psi, dtype=np.float64) / psi  # [psi]

    var_scale = tf.Variable(1.0, dtype=np.float64)

    loss_fn = lambda: -tf.reduce_mean(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=const_probs),
            components_distribution=tfd.Independent(
                tfd.Normal(loc=const_locs, scale=const_scale_base * var_scale),
                reinterpreted_batch_ndims=1,
            ),
        ).log_prob(X)
    )

    loss = loss_fn().numpy()

    for epoch in range(n_epochs):
        optimizer.minimize(loss_fn, var_list=[var_scale])
        var_loss = loss_fn()
        if tf.math.is_nan(var_loss):
            break
        loss = var_loss.numpy()
        if print_epochs:
            print("epoch: ", epoch)
            print("loss: ", loss)

    probs = xpUtils.to_numpy(const_probs)
    locs = xpUtils.to_numpy(const_locs)
    scales = xpUtils.to_numpy(const_scale_base * var_scale)

    return probs, locs, scales


def _test_tf_train():
    import tensorflow as tf

    tnp = tf.experimental.numpy

    n_gaussians = 3
    dims = 1
    n = 1000

    mix = gaussian_mixture(n_gaussians, dims, xp=tnp)
    X = mix.sample(n, dims)
    p_true = mix.prob(X)

    with tf.device("/CPU:0"):
        probs, locs, scales = _tf_train(X, psi=4, print_epochs=True)

    # from timeit import timeit

    # with tf.device("/GPU:0"):
    #     t = timeit(lambda: _tf_train(X, psi=4), number=10)
    #     print("gpu: ", t)

    # with tf.device("/CPU:0"):
    #     t = timeit(lambda: _tf_train(X, psi=4), number=10)
    #     print("cpu: ", t)


_gaussian = "gaussian"
_kernels = [_gaussian]


def _batch_gaussian(X, locs, scales):
    xp, xpUtils = get_array_module(X)
    _, dims = X.shape
    scales = xp.expand_dims(scales, axis=0)
    p = xpUtils.pdist(X, locs, root=False)  # [n, psi]
    p = xp.exp(p / (0 - 2 * (scales**2)))
    p = p / scales / xp.sqrt((np.pi * 2) ** dims)
    return p


class FuzzyINNPartitioning(ReservoirSamplingEstimator, TransformerMixin):
    def __init__(self, psi, kernel=_gaussian):
        super().__init__(psi)
        if kernel not in _kernels:
            raise Exception("This kernel is not supported.")
        self.kernel = kernel

    def partial_fit(self, X, y=None):
        super().partial_fit(X, y)
        return self

    def transform(self, X):
        xp, xpUtils = get_array_module(X)
        if self.kernel == _gaussian:
            locs = self.samples_  # [psi, dims]
            scales = xp.sort(xpUtils.pdist(locs, locs), axis=1)[:, 1]  # [psi]
            return _batch_gaussian(X, locs, scales)
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    _test_tf_train()
