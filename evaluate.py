import numpy as np

h = 28
n = 1885

def rmsse(actual, prediction, train_series, axis=1):
    assert axis == 0 or axis == 1
    assert type(actual) == np.ndarray and type(prediction) == np.ndarray and type(train_series) == np.ndarray

    if axis == 1:
        assert actual.shape[1] > 1 and prediction.shape[1] > 1 and train_series.shape[1] > 1

    numerator = ((actual - prediction) ** 2).sum(axis=axis)
    if axis == 1:
        denominator = 1 / (n - 1) * ((train_series[:, 1:] - train_series[:, :-1]) ** 2).sum(axis=axis)
    else:
        denominator = 1 / (n - 1) * ((train_series[1:] - train_series[:-1]) ** 2).sum(axis=axis)
    return (1 / h * numerator / denominator) ** 0.5