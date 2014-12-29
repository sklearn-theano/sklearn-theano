import warnings
from sklearn.cross_validation import ShuffleSplit
from itertools import chain
from sklearn.utils import safe_indexing
import numpy as np
import scipy.sparse as sp


# A port of sklearn 0.16 utilities
# to avoid validation issues in older sklearn
def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    uniques = np.unique([_num_samples(X) for X in arrays if X is not None])
    if len(uniques) > 1:
        raise ValueError("Found arrays with inconsistent numbers of samples: %s"
                         % str(uniques))


def indexable(*iterables):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """
    result = []
    for X in iterables:
        if sp.issparse(X):
            result.append(X.tocsr())
        elif hasattr(X, "__getitem__") or hasattr(X, "iloc"):
            result.append(X)
        elif X is None:
            result.append(X)
        else:
            result.append(np.array(X))
    check_consistent_length(*result)
    return result


def _num_samples(x):
    """Return number of samples in array-like x."""
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %r" % x)
    return x.shape[0] if hasattr(x, 'shape') else len(x)


def train_test_split(*arrays, **options):
    """Split arrays or matrices into random train and test subsets

    Quick utility that wraps input validation and
    ``next(iter(ShuffleSplit(n_samples)))`` and application to input
    data into a single call for splitting (and optionally subsampling)
    data in a oneliner.

    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
        Python lists or tuples occurring in arrays are converted to 1D numpy
        arrays.

    test_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    Returns
    -------
    splitting : list of arrays, length=2 * len(arrays)
        List containing train-test split of input array.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_validation import train_test_split
    >>> a, b = np.arange(10).reshape((5, 2)), range(5)
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(b)
    [0, 1, 2, 3, 4]

    >>> a_train, a_test, b_train, b_test = train_test_split(
    ...     a, b, test_size=0.33, random_state=42)
    ...
    >>> a_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> b_train
    [2, 0, 3]
    >>> a_test
    array([[2, 3],
           [8, 9]])
    >>> b_test
    [1, 4]

    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    dtype = options.pop('dtype', None)
    if dtype is not None:
        warnings.warn("dtype option is ignored and will be removed in 0.18.")

    force_arrays = options.pop('force_arrays', False)
    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))
    if force_arrays:
        warnings.warn("The force_arrays option is deprecated and will be "
                      "removed in sklearn 0.18.", DeprecationWarning)

    if test_size is None and train_size is None:
        test_size = 0.25
    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    cv = ShuffleSplit(n_samples, test_size=test_size,
                      train_size=train_size,
                      random_state=random_state)

    train, test = next(iter(cv))
    return list(chain.from_iterable((safe_indexing(a, train),
                                     safe_indexing(a, test)) for a in arrays))
