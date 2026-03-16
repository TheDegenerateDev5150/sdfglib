import pytest
import numpy as np
import scipy.special

from docc.python import native


def test_softmax_simple():
    @native
    def softmax_simple(
        A,
    ):
        return scipy.special.softmax(A, axis=0)

    A = np.random.rand(10, 10).astype(np.float64)
    A_ = A.copy()
    res = softmax_simple(A)
    res_ = scipy.special.softmax(A_, axis=0)
    assert np.allclose(res, res_)


def test_softmax_all():
    @native
    def softmax_all(
        A,
    ):
        return scipy.special.softmax(A)

    A = np.random.rand(10, 10).astype(np.float64)
    A_ = A.copy()
    res = softmax_all(A)
    res_ = scipy.special.softmax(A_)
    assert np.allclose(res, res_)
