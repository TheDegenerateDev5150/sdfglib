"""Tests for ControlFlowNode base class bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    PrimitiveType,
    DebugInfo,
)


class TestControlFlowNode:
    """Test suite for ControlFlowNode base class properties."""

    def test_element_id(self):
        """Test that element_id is accessible."""
        builder = StructuredSDFGBuilder("test_sdfg")
        sdfg = builder.move()

        root = sdfg.root
        assert root.element_id >= 0

    def test_debug_info(self):
        """Test that debug_info is accessible."""
        builder = StructuredSDFGBuilder("test_sdfg")
        debug_info = DebugInfo("test.py", 1, 0, 10, 0)
        builder.add_block(debug_info)

        sdfg = builder.move()
        block = sdfg.root.child(0)

        info = block.debug_info
        assert info is not None

    def test_repr(self):
        """Test ControlFlowNode string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")
        sdfg = builder.move()

        root = sdfg.root
        repr_str = repr(root)
        assert "Sequence" in repr_str
