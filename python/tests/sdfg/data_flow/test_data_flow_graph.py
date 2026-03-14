"""Tests for DataFlowGraph bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    PrimitiveType,
    TaskletCode,
)


class TestDataFlowGraph:
    """Test suite for DataFlowGraph properties."""

    def test_nodes_property(self):
        """Test that nodes property returns all nodes in the graph."""
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
        nodes = list(block.dataflow.nodes)

        # Should have 2 access nodes + 1 tasklet = 3 nodes
        assert len(nodes) == 3

    def test_edges_property(self):
        """Test that edges property returns all memlets in the graph."""
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

    def test_in_edges(self):
        """Test querying incoming edges for a node."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_container("y", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_container("z", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        x_ptr = builder.add_access(block_ptr, "x")
        y_ptr = builder.add_access(block_ptr, "y")
        z_ptr = builder.add_access(block_ptr, "z")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.fp_add, ["_in0", "_in1"], ["_out"]
        )
        builder.add_memlet(block_ptr, x_ptr, "", tasklet_ptr, "_in0")
        builder.add_memlet(block_ptr, y_ptr, "", tasklet_ptr, "_in1")
        builder.add_memlet(block_ptr, tasklet_ptr, "_out", z_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)
        tasklets = list(block.dataflow.tasklets)

        assert len(tasklets) == 1
        in_edges = list(block.dataflow.in_edges(tasklets[0]))
        assert len(in_edges) == 2

    def test_out_edges(self):
        """Test querying outgoing edges for a node."""
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
        tasklets = list(block.dataflow.tasklets)

        assert len(tasklets) == 1
        out_edges = list(block.dataflow.out_edges(tasklets[0]))
        assert len(out_edges) == 1

    def test_in_degree(self):
        """Test computing in-degree of a node."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_container("y", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_container("z", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        x_ptr = builder.add_access(block_ptr, "x")
        y_ptr = builder.add_access(block_ptr, "y")
        z_ptr = builder.add_access(block_ptr, "z")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.fp_add, ["_in0", "_in1"], ["_out"]
        )
        builder.add_memlet(block_ptr, x_ptr, "", tasklet_ptr, "_in0")
        builder.add_memlet(block_ptr, y_ptr, "", tasklet_ptr, "_in1")
        builder.add_memlet(block_ptr, tasklet_ptr, "_out", z_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)
        tasklets = list(block.dataflow.tasklets)

        assert len(tasklets) == 1
        assert block.dataflow.in_degree(tasklets[0]) == 2

    def test_out_degree(self):
        """Test computing out-degree of a node."""
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
        tasklets = list(block.dataflow.tasklets)

        assert len(tasklets) == 1
        assert block.dataflow.out_degree(tasklets[0]) == 1

    def test_tasklets_property(self):
        """Test tasklets property returns only tasklet nodes."""
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

        tasklets = list(block.dataflow.tasklets)
        assert len(tasklets) == 1

    def test_data_nodes_property(self):
        """Test data_nodes property returns only access nodes."""
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

        data_nodes = list(block.dataflow.data_nodes)
        assert len(data_nodes) == 2

    def test_reads_property(self):
        """Test reads property returns only read access nodes."""
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

        reads = list(block.dataflow.reads)
        assert len(reads) == 1
        assert reads[0].data == "x"

    def test_writes_property(self):
        """Test writes property returns only write access nodes."""
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

        writes = list(block.dataflow.writes)
        assert len(writes) == 1
        assert writes[0].data == "y"

    def test_sources_property(self):
        """Test sources property returns nodes with no incoming edges."""
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

        sources = list(block.dataflow.sources)
        assert len(sources) == 1

    def test_sinks_property(self):
        """Test sinks property returns nodes with no outgoing edges."""
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

        sinks = list(block.dataflow.sinks)
        assert len(sinks) == 1

    def test_predecessors(self):
        """Test querying predecessor nodes."""
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
        tasklets = list(block.dataflow.tasklets)

        preds = list(block.dataflow.predecessors(tasklets[0]))
        assert len(preds) == 1

    def test_successors(self):
        """Test querying successor nodes."""
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
        tasklets = list(block.dataflow.tasklets)

        succs = list(block.dataflow.successors(tasklets[0]))
        assert len(succs) == 1

    def test_topological_sort(self):
        """Test topological sorting of nodes."""
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

        sorted_nodes = list(block.dataflow.topological_sort())
        assert len(sorted_nodes) == 3

        # Verify topological order: x before tasklet, tasklet before y
        node_order = {node.element_id: i for i, node in enumerate(sorted_nodes)}

        x_node = [n for n in sorted_nodes if hasattr(n, "data") and n.data == "x"]
        y_node = [n for n in sorted_nodes if hasattr(n, "data") and n.data == "y"]
        tasklets = list(block.dataflow.tasklets)

        if x_node and y_node and tasklets:
            x_idx = node_order[x_node[0].element_id]
            y_idx = node_order[y_node[0].element_id]
            t_idx = node_order[tasklets[0].element_id]

            assert x_idx < t_idx < y_idx

    def test_repr(self):
        """Test DataFlowGraph string representation."""
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

        repr_str = repr(block.dataflow)
        assert "DataFlowGraph" in repr_str
        assert "nodes=" in repr_str
        assert "edges=" in repr_str
