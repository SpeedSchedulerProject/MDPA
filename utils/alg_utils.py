import numpy as np
import scipy.signal


class MovingAverage(object):

    def __init__(self, decay=0.95):
        self._decay = decay
        self._shadow = {}

    def update(self, name, x):
        if name not in self._shadow:
            self._shadow[name] = x
        else:
            self._shadow[name] = (1 - self._decay) * x + self._decay * self._shadow[name]
        return self._shadow[name]


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i + 1] + gamma^2 * x[i + 2] + ...
    """
    out = scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    return np.expand_dims(out.astype(np.float32), axis=-1)


def truncate_experiences(lst):
    """
    Truncate experience based on a boolean list.

    e.g.,    [True, False, False, True, True, False] -> [0, 3, 4, 6]  (6 is dummy)
    """
    batch_points = [i for i, x in enumerate(lst) if x]
    batch_points.append(len(lst))

    return batch_points
