import numpy as np
from scipy.optimize import linear_sum_assignment
from Common import to_numpy

eps = np.finfo(float).eps

# y_true should not include noise (-1 or 0)
def fmeasure(y_true, y_pred):

    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    y_true, y_pred = _fix_array(y_true, y_pred)

    matrix = np.zeros([np.max(y_true), np.max(y_pred)])
    noise = np.zeros([np.max(y_true), 1])
    n = y_pred.shape[0]
    for i in range(n):
        if y_pred[i, 0] > 0:
            matrix[y_true[i, 0] - 1, y_pred[i, 0] - 1] = (
                matrix[y_true[i, 0] - 1, y_pred[i, 0] - 1] + 1
            )
        else:
            noise[y_true[i] - 1] = noise[y_true[i] - 1] + 1

    indF = np.zeros([np.max(y_true), 1])
    if matrix.shape[1] != 0:
        f1, recall, precision, match = _fmean(matrix, noise)
        posc, _ = np.where(match == 1)
        for i in range(posc.shape[0]):
            indF[posc[i]] = f1[i]
        f1 = np.sum(indF) / np.max(y_true)
        recall = np.sum(recall) / np.max(y_true)
        precision = np.sum(precision) / np.max(y_true)
    else:
        f1 = 0
        recall = 0
        precision = 0

    return f1, recall, precision


def _fix_shape(X):
    X = np.array(X)
    if len(X.shape) == 1:
        return np.array([X]).T
    else:
        return X


def _fix_array(y_true, y_pred):
    y_true = _fix_shape(y_true)
    y_pred = _fix_shape(y_pred)

    if np.any(y_true == 0):
        y_true[y_true >= 0] = y_true[y_true >= 0] + 1

    if np.any(y_pred == 0):
        y_pred[y_pred >= 0] = y_pred[y_pred >= 0] + 1

    if np.all(y_pred == -1):
        y_pred = y_pred + 2
    return y_true, y_pred


def _fmean(matrix, noise):
    # first round
    recall = _calc_recall(matrix)
    precision = _calc_precision(matrix)

    f1 = 2 * precision * recall / (precision + recall + eps)
    match = _hungarian(0 - f1)
    matrix = np.concatenate([matrix, noise], axis=1)
    r, c = matrix.shape
    rr, cc = match.shape
    match1 = np.concatenate([match, np.zeros([r - rr, cc])], axis=0)
    match1 = np.concatenate([match1, np.zeros([r, c - cc])], axis=1)
    # re-calculate
    recall = _calc_recall(matrix)
    precision = _calc_precision(matrix)
    f1 = 2 * precision * recall / (precision + recall + eps)

    f1 = f1[match1 == 1]
    precision = precision[match1 == 1]
    recall = recall[match1 == 1]

    return f1, recall, precision, match


def _hungarian(cost):
    result = np.zeros_like(cost)
    row_ind, col_ind = linear_sum_assignment(cost)
    for idx, i in enumerate(row_ind):
        j = col_ind[idx]
        result[i, j] = 1
    return result


def _calc_recall(matrix):
    m_recall = np.copy(matrix)
    sumrow = np.sum(matrix, axis=1, keepdims=True)
    if matrix.shape[1] == 1:
        sumrow = matrix
    for j in range(matrix.shape[0]):
        # calculate the total positive examples
        m_recall[j, :] = m_recall[j, :] / (sumrow[j, :] + eps)
    return m_recall


def _calc_precision(matrix):
    m_precision = np.copy(matrix)
    sumcol = np.sum(matrix, axis=0, keepdims=True)
    if matrix.shape[0] == 1:
        sumcol = matrix
    for j in range(matrix.shape[1]):
        # calculate the total positive examples
        m_precision[:, j] = m_precision[:, j] / (sumcol[:, j] + eps)
    return m_precision

