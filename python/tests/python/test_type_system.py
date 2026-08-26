import numpy as np
import pytest

from docc.sdfg import Scalar, PrimitiveType as PT
from docc.python import native
from docc.python.type_system import (
    promote_scalar_types,
    promote_element_types,
    numpy_promote_types,
    scalar_type_for_dtype,
    TypeSystem,
    FLOAT_PRIMITIVE_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _float_ptypes(sdfg):
    """Set of floating-point primitive types among the SDFG's transients."""
    result = set()
    for name in sdfg.containers:
        if sdfg.is_argument(name):
            continue
        pt = getattr(sdfg.type(name), "primitive_type", None)
        if pt in FLOAT_PRIMITIVE_TYPES:
            result.add(pt)
    return result


def _f32(n=4):
    return np.arange(1, n + 1, dtype=np.float32)


def _f64(n=4):
    return np.arange(1, n + 1, dtype=np.float64)


# ===========================================================================
# Axioms: pure promotion functions decide each simple case individually.
# ===========================================================================
class TestPromotionAxioms:
    """Each simple operand pair resolves to the NEP 50 result type."""

    # (left_pt, left_weak, right_pt, right_weak, expected_pt)
    WEAK_CASES = [
        # weak float adapts to a concrete float
        (PT.Float, False, PT.Double, True, PT.Float),
        (PT.Double, True, PT.Float, False, PT.Float),
        # weak int adapts to a concrete float
        (PT.Float, False, PT.Int64, True, PT.Float),
        (PT.Int64, True, PT.Float, False, PT.Float),
        # weak float + concrete int -> default float64
        (PT.Double, True, PT.Int64, False, PT.Double),
        (PT.Int64, False, PT.Double, True, PT.Double),
        # weak int adapts to a concrete int
        (PT.Int32, False, PT.Int64, True, PT.Int32),
        (PT.Int64, True, PT.Int32, False, PT.Int32),
        # half / bfloat keep precision against weak literals
        (PT.Half, False, PT.Double, True, PT.Half),
        (PT.BFloat, False, PT.Int64, True, PT.BFloat),
        # two weak operands fall back to Python defaults
        (PT.Double, True, PT.Double, True, PT.Double),
        (PT.Int64, True, PT.Int64, True, PT.Int64),
        (PT.Int64, True, PT.Double, True, PT.Double),
        # two concrete operands fall back to standard promotion
        (PT.Float, False, PT.Float, False, PT.Float),
        (PT.Float, False, PT.Double, False, PT.Double),
        (PT.Float, False, PT.Int64, False, PT.Double),  # concrete int, not weak
    ]

    @pytest.mark.parametrize("lp,lw,rp,rw,exp", WEAK_CASES)
    def test_promote_scalar_types(self, lp, lw, rp, rw, exp):
        got = promote_scalar_types(Scalar(lp), lw, Scalar(rp), rw)
        assert got.primitive_type == exp

    @pytest.mark.parametrize("lp,lw,rp,rw,exp", WEAK_CASES)
    def test_promote_scalar_types_is_commutative(self, lp, lw, rp, rw, exp):
        got = promote_scalar_types(Scalar(rp), rw, Scalar(lp), lw)
        assert got.primitive_type == exp

    # Standard (concrete-concrete) promotion axioms.
    ELEMENT_CASES = [
        (PT.Float, PT.Float, PT.Float),
        (PT.Double, PT.Double, PT.Double),
        (PT.Float, PT.Double, PT.Double),
        (PT.Int64, PT.Int64, PT.Int64),
        (PT.Int32, PT.Int32, PT.Int32),
        (PT.Int32, PT.Int64, PT.Int64),
        (PT.Float, PT.Int16, PT.Float),  # float32 can represent int16
        (PT.Float, PT.Int32, PT.Double),  # float32 cannot represent int32
        (PT.Float, PT.Int64, PT.Double),
        (PT.Half, PT.Half, PT.Half),
        (PT.BFloat, PT.BFloat, PT.BFloat),
        (PT.Half, PT.Float, PT.Float),
        (PT.Half, PT.Int64, PT.Float),  # half + int -> float32
    ]

    @pytest.mark.parametrize("lp,rp,exp", ELEMENT_CASES)
    def test_promote_element_types(self, lp, rp, exp):
        assert promote_element_types(Scalar(lp), Scalar(rp)).primitive_type == exp

    def test_numpy_scalar_adapts_to_array(self):
        # A scalar (not-array) adapts to the array operand's dtype.
        assert (
            numpy_promote_types(Scalar(PT.Float), True, Scalar(PT.Double), False)
        ).primitive_type == PT.Float
        assert (
            numpy_promote_types(Scalar(PT.Double), False, Scalar(PT.Float), True)
        ).primitive_type == PT.Float


# ===========================================================================
# Base case: a single binary op preserves / promotes container types.
# ===========================================================================
class TestSingleOpBaseCase:
    """One scalar-path binary op yields the expected element type."""

    def test_float32_times_float_literal_stays_float32(self):
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = A[i] * 2.0

        sdfg = kernel.to_sdfg(_f32(), np.zeros(4, np.float32))
        assert _float_ptypes(sdfg) == {PT.Float}

    def test_float32_times_int_literal_stays_float32(self):
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = A[i] * 2

        sdfg = kernel.to_sdfg(_f32(), np.zeros(4, np.float32))
        assert _float_ptypes(sdfg) == {PT.Float}

    def test_float32_plus_float_literal_stays_float32(self):
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = A[i] + 1.5

        sdfg = kernel.to_sdfg(_f32(), np.zeros(4, np.float32))
        assert _float_ptypes(sdfg) == {PT.Float}

    def test_float32_element_times_element_stays_float32(self):
        @native
        def kernel(A, B, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = A[i] * B[i]

        sdfg = kernel.to_sdfg(_f32(), _f32(), np.zeros(4, np.float32))
        assert _float_ptypes(sdfg) == {PT.Float}

    def test_float64_literal_stays_double(self):
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = A[i] * 2.0

        sdfg = kernel.to_sdfg(_f64(), np.zeros(4, np.float64))
        assert _float_ptypes(sdfg) == {PT.Double}

    def test_float32_times_concrete_int_promotes_to_double(self):
        # A concrete (non-literal) integer is NOT weak: float32 * int64 -> float64.
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = A[i] * i

        sdfg = kernel.to_sdfg(_f32(), np.zeros(4, np.float64))
        assert PT.Double in _float_ptypes(sdfg)


# ===========================================================================
# Inductive step: nested expressions preserve the invariant.
# ===========================================================================
class TestNestedInduction:
    """If every sub-expression keeps float32, so does their combination.

    Each depth adds one more operation over float32 operands and weak literals;
    the invariant "only Float floats appear, never Double" must persist.
    """

    def test_depth1(self):
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = A[i] * 2.0

        sdfg = kernel.to_sdfg(_f32(), np.zeros(4, np.float32))
        assert _float_ptypes(sdfg) == {PT.Float}

    def test_depth2(self):
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = A[i] * 2.0 + 1.0

        sdfg = kernel.to_sdfg(_f32(), np.zeros(4, np.float32))
        assert _float_ptypes(sdfg) == {PT.Float}

    def test_depth3(self):
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = (A[i] * 2.0 + 1.0) * 3.0

        sdfg = kernel.to_sdfg(_f32(), np.zeros(4, np.float32))
        assert _float_ptypes(sdfg) == {PT.Float}

    def test_depth4_mixed_ops_and_literals(self):
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = ((A[i] * 2.0 + 1.0) * 3.0 - A[i]) / 2.0

        sdfg = kernel.to_sdfg(_f32(), np.zeros(4, np.float32))
        assert _float_ptypes(sdfg) == {PT.Float}

    def test_nested_double_stays_double(self):
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = ((A[i] * 2.0 + 1.0) * 3.0 - A[i]) / 2.0

        sdfg = kernel.to_sdfg(_f64(), np.zeros(4, np.float64))
        assert _float_ptypes(sdfg) == {PT.Double}

    def test_float32_accumulator_seeded_from_element_stays_float32(self):
        # The matmul-style reduction: an accumulator seeded from a typed value
        # (A[i] * 0.0) keeps float32 across the whole reduction.
        @native
        def kernel(A, B, C):
            n = A.shape[0]
            for i in range(n):
                acc = A[i] * 0.0
                for k in range(n):
                    acc = acc + A[k] * B[k]
                C[i] = acc

        sdfg = kernel.to_sdfg(_f32(), _f32(), np.zeros(4, np.float32))
        assert _float_ptypes(sdfg) == {PT.Float}


# ===========================================================================
# End-to-end: the inferred types also produce numerically correct results.
# ===========================================================================
class TestNumericCorrectness:
    def test_float32_literal_expression(self):
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = A[i] * 2.0 + 1.0

        A = _f32()
        C = np.zeros(4, np.float32)
        kernel(A, C)
        assert C.dtype == np.float32
        np.testing.assert_allclose(C, A * 2.0 + 1.0, rtol=1e-6)

    def test_float32_accumulator_matches_dot(self):
        @native
        def kernel(A, B, C):
            n = A.shape[0]
            for i in range(n):
                acc = A[i] * 0.0
                for k in range(n):
                    acc = acc + A[k] * B[k]
                C[i] = acc

        A = _f32()
        B = _f32()
        C = np.zeros(4, np.float32)
        kernel(A, B, C)
        assert C.dtype == np.float32
        np.testing.assert_allclose(C, np.full(4, np.dot(A, B)), rtol=1e-6)


# ===========================================================================
# Central authority: every path decides types via one TypeSystem instance.
# ===========================================================================
class TestCentralTypeSystem:
    """The shared TypeSystem is the single source of truth for scalar types."""

    def test_constant_type_bool_before_int(self):
        # bool is a subclass of int; the shared rule must classify it as Bool.
        ts = TypeSystem({})
        assert ts.constant_type(True).primitive_type == PT.Bool
        assert ts.constant_type(5).primitive_type == PT.Int64
        assert ts.constant_type(2.0).primitive_type == PT.Double

    def test_result_type_matches_binop(self):
        ts = TypeSystem({"x": Scalar(PT.Float)})
        # weak literal adapts to the concrete float32 container
        assert ts.result_type("x", "2.0", "*").primitive_type == PT.Float
        assert ts.result_type("x", "2", "+").primitive_type == PT.Float
        # true division of two ints is float
        assert ts.result_type("3", "2", "/").primitive_type == PT.Double
        # plain int arithmetic stays integer
        assert ts.result_type("3", "2", "+").primitive_type == PT.Int64

    @pytest.mark.parametrize(
        "dtype_like,expected",
        [
            (np.float64, PT.Double),
            (np.float32, PT.Float),
            (np.float16, PT.Half),
            (np.int64, PT.Int64),
            (np.int32, PT.Int32),
            (np.uint8, PT.UInt8),
            (bool, PT.Bool),
            (int, PT.Int64),
            (float, PT.Double),
        ],
    )
    def test_scalar_type_for_dtype_table(self, dtype_like, expected):
        assert scalar_type_for_dtype(dtype_like).primitive_type == expected

    def test_assign_bool_constant_infers_bool(self):
        # Regression: `flag = True` used to be typed Int64 (int checked first).
        @native
        def kernel(C):
            flag = True
            if flag:
                C[0] = 1.0

        sdfg = kernel.to_sdfg(np.zeros(1, np.float32))
        bool_containers = [
            name
            for name in sdfg.containers
            if not sdfg.is_argument(name)
            and getattr(sdfg.type(name), "primitive_type", None) == PT.Bool
        ]
        assert bool_containers

    def test_float32_min_max_stays_float32(self):
        # Regression: min/max only recognised double, so float32 became int.
        @native
        def kernel(A, C):
            n = A.shape[0]
            for i in range(n):
                C[i] = max(A[i], 0.0)

        sdfg = kernel.to_sdfg(_f32(), np.zeros(4, np.float32))
        assert _float_ptypes(sdfg) == {PT.Float}

    def test_array_result_type_scalar_adapts_to_array(self):
        # The numpy handler resolves elementwise result types here.
        ts = TypeSystem({"A": Scalar(PT.Float)}, tensor_table={"A": object()})
        # array + weak literal -> array dtype (scalar adapts)
        assert ts.array_result_type("A", "2.0").primitive_type == PT.Float
        assert ts.array_result_type("2.0", "A").primitive_type == PT.Float

    def test_promote_and_promote_many_array_ops(self):
        # matmul / outer / einsum resolve element types here.
        ts = TypeSystem(
            {"A": Scalar(PT.Float), "B": Scalar(PT.Double), "C": Scalar(PT.Float)},
            tensor_table={"A": object(), "B": object(), "C": object()},
        )
        assert ts.promote("A", "B").primitive_type == PT.Double
        assert ts.promote("A", "C").primitive_type == PT.Float
        assert ts.promote_many(["A", "C", "B"]).primitive_type == PT.Double

    def test_is_array(self):
        ts = TypeSystem({"A": Scalar(PT.Float)}, tensor_table={"A": object()})
        assert ts.is_array("A")
        assert not ts.is_array("2.0")
