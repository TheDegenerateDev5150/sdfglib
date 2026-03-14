"""Tests for CodeNode, Tasklet, and LibraryNode bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    PrimitiveType,
    DebugInfo,
    TaskletCode,
)


class TestCodeNode:
    """Test suite for CodeNode base class properties."""

    def test_inputs_property(self):
        """Test that inputs property returns input connector names."""
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
        inputs = tasklets[0].inputs
        assert len(inputs) == 2
        assert "_in0" in inputs
        assert "_in1" in inputs

    def test_outputs_property(self):
        """Test that outputs property returns output connector names."""
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
        outputs = tasklets[0].outputs
        assert len(outputs) == 1
        assert "_out" in outputs

    def test_input_by_index(self):
        """Test accessing input connector by index."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("a", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_container("b", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_container("c", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        a_ptr = builder.add_access(block_ptr, "a")
        b_ptr = builder.add_access(block_ptr, "b")
        c_ptr = builder.add_access(block_ptr, "c")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.fp_mul, ["_x", "_y"], ["_z"]
        )
        builder.add_memlet(block_ptr, a_ptr, "", tasklet_ptr, "_x")
        builder.add_memlet(block_ptr, b_ptr, "", tasklet_ptr, "_y")
        builder.add_memlet(block_ptr, tasklet_ptr, "_z", c_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)
        tasklets = list(block.dataflow.tasklets)

        assert len(tasklets) == 1
        assert tasklets[0].input(0) == "_x"
        assert tasklets[0].input(1) == "_y"

    def test_output_by_index(self):
        """Test accessing output connector by index."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_container("y", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        x_ptr = builder.add_access(block_ptr, "x")
        y_ptr = builder.add_access(block_ptr, "y")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.assign, ["_in"], ["_result"]
        )
        builder.add_memlet(block_ptr, x_ptr, "", tasklet_ptr, "_in")
        builder.add_memlet(block_ptr, tasklet_ptr, "_result", y_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)
        tasklets = list(block.dataflow.tasklets)

        assert len(tasklets) == 1
        assert tasklets[0].output(0) == "_result"


class TestTasklet:
    """Test suite for Tasklet properties."""

    def test_code_property(self):
        """Test that code property returns the tasklet operation code."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_container("y", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        x_ptr = builder.add_access(block_ptr, "x")
        y_ptr = builder.add_access(block_ptr, "y")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.fp_neg, ["_in"], ["_out"]
        )
        builder.add_memlet(block_ptr, x_ptr, "", tasklet_ptr, "_in")
        builder.add_memlet(block_ptr, tasklet_ptr, "_out", y_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)
        tasklets = list(block.dataflow.tasklets)

        assert len(tasklets) == 1
        assert tasklets[0].code == TaskletCode.fp_neg

    def test_tasklet_repr(self):
        """Test Tasklet string representation."""
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

        repr_str = repr(tasklets[0])
        assert "Tasklet" in repr_str

    def test_various_tasklet_codes(self):
        """Test different tasklet operation codes."""
        # Float operations use Float type, int operations use Int32
        codes_to_test = [
            (TaskletCode.assign, 1, 1, PrimitiveType.Float),
            (TaskletCode.fp_add, 2, 1, PrimitiveType.Float),
            (TaskletCode.fp_sub, 2, 1, PrimitiveType.Float),
            (TaskletCode.fp_mul, 2, 1, PrimitiveType.Float),
            (TaskletCode.fp_div, 2, 1, PrimitiveType.Float),
            (TaskletCode.int_add, 2, 1, PrimitiveType.Int32),
            (TaskletCode.int_mul, 2, 1, PrimitiveType.Int32),
        ]

        for code, num_inputs, num_outputs, ptype in codes_to_test:
            builder = StructuredSDFGBuilder(f"test_{code}")

            # Create containers with appropriate type
            if num_inputs == 1:
                builder.add_container("x", Scalar(ptype), is_argument=True)
                inputs = ["_in"]
            else:
                builder.add_container("x", Scalar(ptype), is_argument=True)
                builder.add_container("y", Scalar(ptype), is_argument=True)
                inputs = ["_in0", "_in1"]

            builder.add_container("z", Scalar(ptype), is_argument=True)
            outputs = ["_out"]

            block_ptr = builder.add_block()
            if num_inputs == 1:
                x_ptr = builder.add_access(block_ptr, "x")
            else:
                x_ptr = builder.add_access(block_ptr, "x")
                y_ptr = builder.add_access(block_ptr, "y")

            z_ptr = builder.add_access(block_ptr, "z")
            tasklet_ptr = builder.add_tasklet(block_ptr, code, inputs, outputs)

            if num_inputs == 1:
                builder.add_memlet(block_ptr, x_ptr, "", tasklet_ptr, "_in")
            else:
                builder.add_memlet(block_ptr, x_ptr, "", tasklet_ptr, "_in0")
                builder.add_memlet(block_ptr, y_ptr, "", tasklet_ptr, "_in1")

            builder.add_memlet(block_ptr, tasklet_ptr, "_out", z_ptr, "")

            sdfg = builder.move()
            block = sdfg.root.child(0)
            tasklets = list(block.dataflow.tasklets)

            assert len(tasklets) == 1
            assert tasklets[0].code == code
            assert len(tasklets[0].inputs) == num_inputs
            assert len(tasklets[0].outputs) == num_outputs


class TestLibraryNode:
    """Test suite for LibraryNode properties."""

    def test_library_node_in_graph(self):
        """Test that library nodes can be queried from the dataflow graph."""
        # Note: This test may need adjustment based on how library nodes are created
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        builder.add_access(block_ptr, "x")

        sdfg = builder.move()
        block = sdfg.root.child(0)

        # Library nodes list should be accessible even if empty
        lib_nodes = list(block.dataflow.library_nodes)
        assert isinstance(lib_nodes, list)
