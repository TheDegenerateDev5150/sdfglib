"""Tests for DataFlowNode, AccessNode, and ConstantNode bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    PrimitiveType,
    Array,
    DebugInfo,
    TaskletCode,
)


class TestDataFlowNode:
    """Test suite for DataFlowNode base class properties."""

    def test_element_id(self):
        """Test that element_id is accessible and unique."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        access_ptr = builder.add_access(block_ptr, "x")

        sdfg = builder.move()
        root = sdfg.root
        block = root.child(0)
        nodes = list(block.dataflow.nodes)

        assert len(nodes) == 1
        assert nodes[0].element_id >= 0

    def test_debug_info(self):
        """Test that debug_info is accessible."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)

        debug_info = DebugInfo("test.py", 10, 5, 10, 15)
        block_ptr = builder.add_block(debug_info)
        builder.add_access(block_ptr, "x", debug_info)

        sdfg = builder.move()
        root = sdfg.root
        block = root.child(0)
        nodes = list(block.dataflow.nodes)

        assert len(nodes) == 1
        # Debug info should be accessible
        info = nodes[0].debug_info
        assert info is not None


class TestAccessNode:
    """Test suite for AccessNode properties."""

    def test_data_property(self):
        """Test that data property returns the container name."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("my_var", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        builder.add_access(block_ptr, "my_var")

        sdfg = builder.move()
        block = sdfg.root.child(0)
        nodes = list(block.dataflow.data_nodes)

        assert len(nodes) == 1
        assert nodes[0].data == "my_var"

    def test_repr(self):
        """Test AccessNode string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        builder.add_access(block_ptr, "x")

        sdfg = builder.move()
        block = sdfg.root.child(0)
        nodes = list(block.dataflow.data_nodes)

        repr_str = repr(nodes[0])
        assert "AccessNode" in repr_str
        assert "x" in repr_str

    def test_side_effect_read(self):
        """Test that read access nodes have no side effects."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_container("y", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        x_ptr = builder.add_access(block_ptr, "x")
        y_ptr = builder.add_access(block_ptr, "y")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.assign, ["_in"], ["_out"]
        )
        builder.add_memlet(block_ptr, x_ptr, "", tasklet_ptr, "_in")
        builder.add_memlet(block_ptr, tasklet_ptr, "_out", y_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)

        # Find read and write nodes
        reads = list(block.dataflow.reads)
        writes = list(block.dataflow.writes)

        assert len(reads) == 1
        assert len(writes) == 1
        assert reads[0].side_effect is False
        assert writes[0].side_effect is True


class TestConstantNode:
    """Test suite for ConstantNode properties."""

    def test_constant_value(self):
        """Test that constant nodes expose their value."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("y", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        const_ptr = builder.add_constant(block_ptr, "3.14", Scalar(PrimitiveType.Float))
        y_ptr = builder.add_access(block_ptr, "y")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.assign, ["_in"], ["_out"]
        )
        builder.add_memlet(block_ptr, const_ptr, "", tasklet_ptr, "_in")
        builder.add_memlet(block_ptr, tasklet_ptr, "_out", y_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)
        nodes = list(block.dataflow.nodes)

        # Find constant node
        const_nodes = [n for n in nodes if hasattr(n, "type") and "3.14" in n.data]
        assert len(const_nodes) == 1
        assert const_nodes[0].data == "3.14"

    def test_constant_type(self):
        """Test that constant nodes have type property."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("y", Scalar(PrimitiveType.Int32), is_argument=True)

        block_ptr = builder.add_block()
        const_ptr = builder.add_constant(block_ptr, "42", Scalar(PrimitiveType.Int32))
        y_ptr = builder.add_access(block_ptr, "y")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.assign, ["_in"], ["_out"]
        )
        builder.add_memlet(block_ptr, const_ptr, "", tasklet_ptr, "_in")
        builder.add_memlet(block_ptr, tasklet_ptr, "_out", y_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)
        nodes = list(block.dataflow.nodes)

        # Find constant node (has type attribute)
        from docc.sdfg import ConstantNode

        const_nodes = [n for n in nodes if isinstance(n, ConstantNode)]
        assert len(const_nodes) == 1
        assert const_nodes[0].type is not None

    def test_constant_repr(self):
        """Test ConstantNode string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("y", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        const_ptr = builder.add_constant(
            block_ptr, "2.718", Scalar(PrimitiveType.Float)
        )
        y_ptr = builder.add_access(block_ptr, "y")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.assign, ["_in"], ["_out"]
        )
        builder.add_memlet(block_ptr, const_ptr, "", tasklet_ptr, "_in")
        builder.add_memlet(block_ptr, tasklet_ptr, "_out", y_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)

        from docc.sdfg import ConstantNode

        nodes = list(block.dataflow.nodes)
        const_nodes = [n for n in nodes if isinstance(n, ConstantNode)]

        assert len(const_nodes) == 1
        repr_str = repr(const_nodes[0])
        assert "ConstantNode" in repr_str
        assert "2.718" in repr_str
