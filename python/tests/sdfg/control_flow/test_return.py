"""Tests for Return bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    PrimitiveType,
    DebugInfo,
    Return,
)


class TestReturn:
    """Test suite for Return properties."""

    def test_return_isinstance(self):
        """Test that return is an instance of Return."""
        builder = StructuredSDFGBuilder("test_sdfg", Scalar(PrimitiveType.Float))
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_return("x")

        sdfg = builder.move()
        return_node = sdfg.root.child(0)

        assert isinstance(return_node, Return)

    def test_data_property(self):
        """Test data property returns the container name."""
        builder = StructuredSDFGBuilder("test_sdfg", Scalar(PrimitiveType.Float))
        builder.add_container("result", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_return("result")

        sdfg = builder.move()
        return_node = sdfg.root.child(0)

        assert return_node.data == "result"

    def test_constant_return_data(self):
        """Test data property for constant return."""
        builder = StructuredSDFGBuilder("test_sdfg", Scalar(PrimitiveType.Int32))
        builder.add_constant_return("42", Scalar(PrimitiveType.Int32))

        sdfg = builder.move()
        return_node = sdfg.root.child(0)

        assert return_node.data == "42"

    def test_type_property(self):
        """Test type property access - may be None for inferred types."""
        builder = StructuredSDFGBuilder("test_sdfg", Scalar(PrimitiveType.Float))
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_return("x")

        sdfg = builder.move()
        return_node = sdfg.root.child(0)

        # Type property is accessible (may be None if type is inferred)
        _ = return_node.type

    def test_is_data_property(self):
        """Test is_data property for data return."""
        builder = StructuredSDFGBuilder("test_sdfg", Scalar(PrimitiveType.Float))
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_return("x")

        sdfg = builder.move()
        return_node = sdfg.root.child(0)

        assert return_node.is_data is True
        assert return_node.is_constant is False

    def test_is_constant_property(self):
        """Test is_constant property for constant return."""
        builder = StructuredSDFGBuilder("test_sdfg", Scalar(PrimitiveType.Int32))
        builder.add_constant_return("42", Scalar(PrimitiveType.Int32))

        sdfg = builder.move()
        return_node = sdfg.root.child(0)

        assert return_node.is_constant is True
        assert return_node.is_data is False

    def test_return_repr(self):
        """Test Return string representation."""
        builder = StructuredSDFGBuilder("test_sdfg", Scalar(PrimitiveType.Float))
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_return("x")

        sdfg = builder.move()
        return_node = sdfg.root.child(0)

        repr_str = repr(return_node)
        assert "Return" in repr_str
        assert "data=" in repr_str
        assert "is_constant=" in repr_str

    def test_return_element_id(self):
        """Test that element_id is accessible."""
        builder = StructuredSDFGBuilder("test_sdfg", Scalar(PrimitiveType.Float))
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_return("x")

        sdfg = builder.move()
        return_node = sdfg.root.child(0)

        assert return_node.element_id >= 0

    def test_constant_return_float(self):
        """Test constant return with float value."""
        builder = StructuredSDFGBuilder("test_sdfg", Scalar(PrimitiveType.Double))
        builder.add_constant_return("3.14159", Scalar(PrimitiveType.Double))

        sdfg = builder.move()
        return_node = sdfg.root.child(0)

        assert return_node.data == "3.14159"
        assert return_node.is_constant is True

    def test_return_in_conditional(self):
        """Test return inside conditional branch."""
        builder = StructuredSDFGBuilder("test_sdfg", Scalar(PrimitiveType.Int32))
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_if("x > 0")
        builder.add_constant_return("1", Scalar(PrimitiveType.Int32))
        builder.begin_else()
        builder.add_constant_return("0", Scalar(PrimitiveType.Int32))
        builder.end_if()

        sdfg = builder.move()
        ifelse = sdfg.root.child(0)

        then_case = ifelse.case(0)
        else_case = ifelse.case(1)

        then_return = then_case.child(0)
        else_return = else_case.child(0)

        assert isinstance(then_return, Return)
        assert isinstance(else_return, Return)
        assert then_return.data == "1"
        assert else_return.data == "0"
