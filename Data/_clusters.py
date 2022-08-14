import numpy as np
import h5py
import scipy


def load_mat(folder, dataset_name, xp=np):
    file_name = folder + "/" + dataset_name + ".mat"
    try:
        f = h5py.File(file_name, "r")
    except:
        f = scipy.io.loadmat(file_name)
    X = xp.array(f.get("data"))
    y = xp.array(f.get("class"))
    if len(y.shape) == 2:
        if y.shape[1] == 1:
            y = xp.squeeze(y, axis=1)
        elif y.shape[0] == 1:
            y = xp.squeeze(y, axis=0)
    if X.shape[1] == y.shape[0]:
        X = xp.transpose(X)
    # labels = xp.sort(xp.unique(y))
    return X, y
