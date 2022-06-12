# pyright: reportMissingImports=false


def setup_tf(xpUtils):
    import tensorflow as tf
    from scipy import sparse

    xp = tf.experimental.numpy

    def _coo_matrix(X: tf.Tensor, shape=None, dtype=None, copy=False):
        return

    def _hstack(blocks, format=None, dtype=None):
        return sparse.hstack(blocks, format, dtype)

    def _unique(X):
        if len(X.shape) == 2:
            X_, _ = tf.raw_ops.UniqueV2(x=X, axis=[0])
        elif len(X.shape) == 1:
            X_, _ = tf.unique(X)
        return X_

    def _tensor_scatter_nd_update(X, indices, updates):
        if len(indices.shape) == 1:
            indices = xp.expand_dims(indices, axis=1)

        return tf.tensor_scatter_nd_update(X, indices, updates)

    setattr(xpUtils, "coo_matrix", _coo_matrix)
    setattr(xpUtils, "hstack", _hstack)
    setattr(xpUtils, "norm", tf.norm)
    setattr(xpUtils, "to_numpy", lambda X: X.numpy())
    setattr(xpUtils, "unique", lambda X: _unique(X))
    setattr(xpUtils, "copy", lambda X: tf.identity(X))
    setattr(xpUtils, "tile", tf.tile)
    setattr(xpUtils, "cast", tf.cast)
    setattr(xpUtils, "numpy_dtype", lambda X: X.dtype.as_numpy_dtype)
    setattr(xpUtils, "tensor_scatter_nd_update", _tensor_scatter_nd_update)
    setattr(xpUtils, "gather_nd", tf.gather_nd)
    return xp
