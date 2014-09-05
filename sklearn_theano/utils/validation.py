"""Utilities for input validation"""
# License: BSD 3 clause
import numpy as np


def check_tensor(array, dtype=None, order=None, n_dim=None, copy=False):
    """Input validation on an array, or list.

    By default, the input is converted to an at least 2nd numpy array.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    dtype : object
        Input type to check / convert.

    n_dim : int
        Number of dimensions for input array. If smaller, input array will be
        appended by dimensions of length 1 until n_dims is matched.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    array = np.array(array, dtype=dtype, order=order, copy=copy)
    if n_dim is not None:
        if len(array.shape) > n_dim:
            raise ValueError("Input array has shape %s, expected array with "
                             "%s dimensions or less" % (array.shape, n_dim))
        elif len(array.shape) < n_dim:
            array = array[[np.newaxis] * (n_dim - len(array.shape))]
    return array
