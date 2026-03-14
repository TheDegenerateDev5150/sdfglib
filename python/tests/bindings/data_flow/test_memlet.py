"""Tests for Memlet bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    Array,
    PrimitiveType,
    TaskletCode,
    MemletType,
)


class TestMemlet:
    """Test suite for Memlet properties."""

    def test_memlet_type_computational(self):
        """Test that computational memlets have the correct type."""
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
        edges = list(block.dataflow.edges)

        assert len(edges) == 2
        for edge in edges:
            assert edge.type == MemletType.Computational

    def test_memlet_src_dst(self):
        """Test that src and dst properties return the correct nodes."""
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
        edges = list(block.dataflow.edges)

        assert len(edges) == 2
        for edge in edges:
            assert edge.src is not None
            assert edge.dst is not None

    def test_memlet_connectors(self):
        """Test that src_conn and dst_conn are accessible."""
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
        edges = list(block.dataflow.edges)

        # Find edge to tasklet and edge from tasklet
        to_tasklet = [e for e in edges if e.dst_conn == "_in"]
        from_tasklet = [e for e in edges if e.src_conn == "_out"]

        assert len(to_tasklet) == 1
        assert len(from_tasklet) == 1
        assert to_tasklet[0].dst_conn == "_in"
        assert from_tasklet[0].src_conn == "_out"

    def test_memlet_subset_scalar(self):
        """Test that subset is empty for scalar access."""
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
        edges = list(block.dataflow.edges)

        for edge in edges:
            subset = edge.subset
            assert isinstance(subset, list)

    def test_memlet_subset_array(self):
        """Test that subset contains index expressions for array access."""
        builder = StructuredSDFGBuilder("test_sdfg")
        # Array takes element type and size as string
        builder.add_container(
            "arr", Array(Scalar(PrimitiveType.Float), "10"), is_argument=True
        )
        builder.add_container("y", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        arr_ptr = builder.add_access(block_ptr, "arr")
        y_ptr = builder.add_access(block_ptr, "y")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.assign, ["_in"], ["_out"]
        )
        builder.add_memlet(block_ptr, arr_ptr, "", tasklet_ptr, "_in", subset="0")
        builder.add_memlet(block_ptr, tasklet_ptr, "_out", y_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)
        edges = list(block.dataflow.edges)

        # Find the edge from array
        arr_edges = [e for e in edges if e.dst_conn == "_in"]
        assert len(arr_edges) == 1
        subset = arr_edges[0].subset
        assert isinstance(subset, list)
        assert len(subset) == 1  # One dimension
        assert "0" in subset[0]

    def test_memlet_base_type(self):
        """Test that base_type is accessible."""
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
        edges = list(block.dataflow.edges)

        for edge in edges:
            base_type = edge.base_type
            assert base_type is not None

    def test_memlet_element_id(self):
        """Test that element_id is accessible."""
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
        edges = list(block.dataflow.edges)

        for edge in edges:
            assert edge.element_id >= 0

    def test_memlet_repr(self):
        """Test Memlet string representation."""
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
        edges = list(block.dataflow.edges)

        for edge in edges:
            repr_str = repr(edge)
            assert "Memlet" in repr_str


class TestMemletType:
    """Test suite for MemletType enum."""

    def test_memlet_type_values(self):
        """Test that MemletType enum values are accessible."""
        assert MemletType.Computational is not None
        assert MemletType.Reference is not None
        assert MemletType.Dereference_Src is not None
        assert MemletType.Dereference_Dst is not None

    def test_memlet_type_comparison(self):
        """Test MemletType comparison."""
        assert MemletType.Computational == MemletType.Computational
        assert MemletType.Computational != MemletType.Reference
