import os
import numpy as np


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
