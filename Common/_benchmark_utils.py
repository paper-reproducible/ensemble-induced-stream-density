import os
import h5py
import numpy as np


# def load_mat(folder, dataset_name, xp=np, ):
#     file_name = folder + "/" + dataset_name + ".mat"
#     f = h5py.File(file_name, "r")
#     X = xp.array(f.get("X"))
#     y = xp.array(f.get("y"))
#     if len(y.shape) == 2:
#         if y.shape[1] == 1:
#             y = xp.squeeze(y, axis=1)
#         elif y.shape[0] == 1:
#             y = xp.squeeze(y, axis=0)
#     if X.shape[1] == y.shape[0]:
#         X = xp.transpose(X)
#     labels = xp.sort(xp.unique(y))
#     if labels.shape[0] == 2 and labels[0] == 0 and labels[1] == 1:
#         y = xp.where(y == 1, -1, y)
#         y = xp.where(y == 0, 1, y)
#     elif xp.min(labels) < 0:
#         y = xp.where(y >= 0, 1, y)
#         y = xp.where(y < 0, -1, y)
#     else:
#         raise Exception("Unsupported label format")
#     return X, y


def save_parquet(df, file_name):
    ext = ".parquet.gzip"
    file_name = file_name if file_name.endswith(ext) else file_name + ext
    df.to_parquet(file_name, compression="gzip")
    return df, file_name


def save_csv(df, file_name):
    ext = ".csv"
    file_name = file_name if file_name.endswith(ext) else file_name + ext
    df.to_csv(file_name)
    return df, file_name


def init_xp(use_tensorflow=False, use_cupy=False):
    xp = np
    if use_tensorflow:
        import tensorflow as tf

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
        tnp = tf.experimental.numpy
        tnp.experimental_enable_numpy_behavior()
        xp = tnp
    elif use_cupy:
        import cupy as cp

        xp = cp

    return xp
