"""Tests for IfElse bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    PrimitiveType,
    DebugInfo,
    IfElse,
    Sequence,
)


class TestIfElse:
    """Test suite for IfElse properties."""

    def test_ifelse_isinstance(self):
        """Test that if-else node is an instance of IfElse."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        assert isinstance(ifelse, IfElse)

    def test_size_property(self):
        """Test size property returns number of cases."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        assert ifelse.size == 1

    def test_size_with_else(self):
        """Test size with if-else (two cases)."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.begin_else()
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        assert ifelse.size == 2

    def test_case_method(self):
        """Test case() method returns case sequence at index."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        case = ifelse.case(0)
        assert isinstance(case, Sequence)

    def test_condition_method(self):
        """Test condition() method returns condition string at index."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        condition = ifelse.condition(0)
        assert isinstance(condition, str)
        assert "x" in condition

    def test_cases_method(self):
        """Test cases() method returns all case sequences."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.begin_else()
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        cases = ifelse.cases()
        assert len(cases) == 2
        assert all(isinstance(c, Sequence) for c in cases)

    def test_conditions_method(self):
        """Test conditions() method returns all condition strings."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.begin_else()
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        conditions = ifelse.conditions()
        assert len(conditions) == 2
        assert all(isinstance(c, str) for c in conditions)

    def test_is_complete_with_else(self):
        """Test is_complete property for if-else (should be complete)."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.begin_else()
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        # If-else with complementary conditions should be complete
        assert ifelse.is_complete is True

    def test_is_complete_without_else(self):
        """Test is_complete property for if without else."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        # Single if without else is incomplete
        assert ifelse.is_complete is False

    def test_len_dunder(self):
        """Test __len__ returns number of cases."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.begin_else()
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        assert len(ifelse) == 2

    def test_ifelse_repr(self):
        """Test IfElse string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        repr_str = repr(ifelse)
        assert "IfElse" in repr_str
        assert "cases=" in repr_str
        assert "is_complete=" in repr_str

    def test_nested_ifelse(self):
        """Test nested if-else structures."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)
        builder.add_container("y", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        # Nested if
        builder.begin_if("y > 0")
        builder.add_block()
        builder.end_if()
        builder.end_if()

        sdfg = builder.move()
        outer_ifelse = sdfg.root.child(0)

        assert isinstance(outer_ifelse, IfElse)

        # Get the nested if-else from the then branch
        then_case = outer_ifelse.case(0)
        assert then_case.size == 1

        nested_ifelse = then_case.child(0)
        assert isinstance(nested_ifelse, IfElse)

    def test_ifelse_element_id(self):
        """Test that element_id is accessible."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_block()
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        assert ifelse.element_id >= 0
