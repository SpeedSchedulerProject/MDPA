import numpy as np

from tf_compat import tf


class SparseMat(object):

    def __init__(self, dtype, shape, row=None, col=None, data=None):
        self._dtype = dtype
        self._shape = shape
        self._row = row if row is not None else []
        self._col = col if col is not None else []
        self._data = data if data is not None else []

    def add(self, row, col, data):
        self._row.append(row)
        self._col.append(col)
        self._data.append(data)

    def get_col(self):
        return np.asarray(self._col, dtype=np.int32)

    def get_row(self):
        return np.asarray(self._row, dtype=np.int32)

    def get_data(self):
        return np.asarray(self._data, dtype=self._dtype)

    @property
    def shape(self):
        return self._shape

    def to_tf_sp_tensor_value_format(self):
        row_idx = self.get_row()
        col_idx = self.get_col()

        indices = np.vstack([row_idx, col_idx]).transpose()
        data = self.get_data()
        shape = self._shape
        return indices, data, shape

    def to_tf_sp_tensor_value(self):
        indices, data, shape = self.to_tf_sp_tensor_value_format()
        return tf.SparseTensorValue(indices, data, shape)


def expand_sp_mat(sp_mat, batch_size):
    """
    Make a stack of same sparse matrix to a giant one on its diagonal.

    The input is SparseMat

    e.g., expand a sparse matrix batch_size times
    [0, 1, 0]    
    ..  ..   [1, 0, 0]
    ..  ..   ..  ..   [0, 0, 1]
    to
    [0, 1, 0]    
    ..  ..   [1, 0, 0]
    ..  ..   ..  ..   [0, 0, 1]
            ****        batch_size times        ****
    ..  ..   ..  ..   ..  ..   [0, 1, 0]    
    ..  ..   ..  ..   ..  ..   ..  ..   [1, 0, 0]
    ..  ..   ..  ..   ..  ..   ..  ..   ..  ..   [0, 0, 1]
    where ".." are all zeros
    depth is on the 3rd dimension, which is orthogonal to the planar 
    """

    base = 0
    row_idx = []
    col_idx = []
    data = []
    for _ in range(batch_size):
        row_idx.append(sp_mat.get_row() + base)
        col_idx.append(sp_mat.get_col() + base)
        data.append(sp_mat.get_data())
        base += sp_mat.shape[0]

    row_idx = np.hstack(row_idx)
    col_idx = np.hstack(col_idx)
    data = np.hstack(data)

    expanded_sp_mat = SparseMat(dtype=np.float32, shape=[base, base], row=row_idx, col=col_idx, data=data)
    return expanded_sp_mat.to_tf_sp_tensor_value_format()


def expand_sp_mats(sp_mats, batch_size):
    expanded_sp_mats = []
    for sp_mat in sp_mats:
        expanded_sp_mat = expand_sp_mat(sp_mat, batch_size)
        expanded_sp_mats.append(expanded_sp_mat)
    return expanded_sp_mats
