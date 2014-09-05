"""Tests for input validation functions."""
import numpy as np
from sklearn_theano.utils import check_tensor
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises


def test_check_tensor():
    """Test that check_tensor works for a variety of inputs."""
    X = np.zeros((3, 4, 5))
    assert_equal(check_tensor([1, 2]).shape, (2, ))
    assert_raises(ValueError, check_tensor, X, dtype=np.float, n_dim=1)
    assert_equal(check_tensor(X, dtype=np.float, n_dim=6).shape,
                 (1, 1, 1, 3, 4, 5))
    assert_equal(check_tensor(X, dtype=np.float, n_dim=3).shape, (3, 4, 5))
