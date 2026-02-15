from docc.python import native
import numpy as np
import pytest


def test_transpose_T():
    """Test .T attribute transpose with shape/stride checks"""

    # Test 2D C-order .T
    @native
    def transpose_T_2d_c(a):
        return a.T

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = transpose_T_2d_c(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.strides == (8, 24)  # Swapped input strides (view semantics)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D F-order .T
    @native
    def transpose_T_2d_f(a):
        return a.T

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    result = transpose_T_2d_f(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.strides == (16, 8)  # Swapped input strides (view semantics)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer array .T
    @native
    def transpose_T_int(a):
        return a.T

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    result = transpose_T_int(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.strides == (8, 24)  # Swapped input strides
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test square matrix .T
    @native
    def transpose_T_square(a):
        return a.T

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    result = transpose_T_square(a)
    expected = a.T
    assert result.shape == (3, 3)
    assert result.strides == (8, 24)  # Swapped input strides
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    @native
    def transpose_float32(a):
        return a.T

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = transpose_float32(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.strides == (4, 12)  # Swapped input strides for float32
    assert result.dtype == np.float32
    assert np.array_equal(result, expected)

    @native
    def transpose_int32(a):
        return a.T

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    result = transpose_int32(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.strides == (4, 12)  # Swapped input strides for int32
    assert result.dtype == np.int32
    assert np.array_equal(result, expected)


def test_transpose_func():
    """Test np.transpose() function with shape/stride checks"""

    # Test 2D C-order transpose
    @native
    def transpose_func_2d_c(a):
        return np.transpose(a)

    a = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64, order="C"
    )
    result = transpose_func_2d_c(a)
    expected = np.transpose(a)
    assert result.shape == (4, 2)
    assert result.strides == (8, 32)  # Swapped input strides
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D F-order transpose
    @native
    def transpose_func_2d_f(a):
        return np.transpose(a)

    a = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64, order="F"
    )
    result = transpose_func_2d_f(a)
    expected = np.transpose(a)
    assert result.shape == (4, 2)
    assert result.strides == (16, 8)  # Swapped input strides
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer transpose
    @native
    def transpose_func_int(a):
        return np.transpose(a)

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    result = transpose_func_int(a)
    expected = np.transpose(a)
    assert result.shape == (3, 3)
    assert result.strides == (8, 24)  # Swapped input strides
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)


def test_transpose_axes():
    """Test np.transpose() with explicit axes parameter"""

    # Test 2D transpose with axes=(1, 0) - equivalent to .T
    @native
    def transpose_axes_2d(a):
        return np.transpose(a, axes=(1, 0))

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = transpose_axes_2d(a)
    expected = np.transpose(a, axes=(1, 0))
    assert result.shape == (3, 2)
    assert result.strides == (8, 24)  # Swapped input strides
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D identity permutation axes=(0, 1)
    @native
    def transpose_axes_identity(a):
        return np.transpose(a, axes=(0, 1))

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = transpose_axes_identity(a)
    expected = np.transpose(a, axes=(0, 1))
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 3D transpose with axes=(2, 0, 1)
    @native
    def transpose_axes_3d(a):
        return np.transpose(a, axes=(2, 0, 1))

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = transpose_axes_3d(a)
    expected = np.transpose(a, axes=(2, 0, 1))
    assert result.shape == (4, 2, 3)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 3D transpose with axes=(1, 2, 0)
    @native
    def transpose_axes_3d_alt(a):
        return np.transpose(a, axes=(1, 2, 0))

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = transpose_axes_3d_alt(a)
    expected = np.transpose(a, axes=(1, 2, 0))
    assert result.shape == (3, 4, 2)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_transpose_3d():
    """Test 3D array transpose with shape/stride checks"""

    # Test 3D C-order .T (reverses all axes)
    @native
    def transpose_3d_T(a):
        return a.T

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = transpose_3d_T(a)
    expected = a.T
    assert result.shape == (4, 3, 2)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 3D F-order .T
    @native
    def transpose_3d_T_f(a):
        return a.T

    a = np.asfortranarray(np.arange(24, dtype=np.float64).reshape(2, 3, 4))
    result = transpose_3d_T_f(a)
    expected = a.T
    assert result.shape == (4, 3, 2)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 3D transpose preserves values correctly
    @native
    def transpose_3d_check_values(a):
        return np.transpose(a)

    a = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    result = transpose_3d_check_values(a)
    expected = np.transpose(a)
    assert result.shape == (5, 4, 3)
    assert result.dtype == np.float64
    # Check specific values
    assert result[0, 0, 0] == a[0, 0, 0]
    assert result[1, 0, 0] == a[0, 0, 1]
    assert result[0, 1, 0] == a[0, 1, 0]
    assert result[0, 0, 1] == a[1, 0, 0]
    assert np.array_equal(result, expected)


# def test_transpose_strided():
#     """Test transpose with strided (non-contiguous) input arrays"""

#     # Test strided 2D input transpose
#     @native
#     def transpose_strided_2d(a):
#         return a.T

#     a_full = np.arange(24, dtype=np.float64).reshape(4, 6)
#     a = a_full[::2, ::2]  # Shape (2, 3), non-contiguous
#     assert not a.flags['C_CONTIGUOUS']
#     result = transpose_strided_2d(a)
#     expected = a.T
#     assert result.shape == (3, 2)
#     assert result.dtype == np.float64
#     assert np.array_equal(result, expected)

#     # Test row-sliced input transpose
#     @native
#     def transpose_row_sliced(a):
#         return np.transpose(a)

#     a_full = np.arange(24, dtype=np.float64).reshape(6, 4)
#     a = a_full[::2, :]  # Every other row
#     result = transpose_row_sliced(a)
#     expected = np.transpose(a)
#     assert result.shape == (4, 3)
#     assert result.dtype == np.float64
#     assert np.array_equal(result, expected)

#     # Test column-sliced input transpose
#     @native
#     def transpose_col_sliced(a):
#         return np.transpose(a)

#     a_full = np.arange(24, dtype=np.float64).reshape(4, 6)
#     a = a_full[:, ::3]  # Every 3rd column
#     result = transpose_col_sliced(a)
#     expected = np.transpose(a)
#     assert result.shape == (2, 4)
#     assert result.dtype == np.float64
#     assert np.array_equal(result, expected)
