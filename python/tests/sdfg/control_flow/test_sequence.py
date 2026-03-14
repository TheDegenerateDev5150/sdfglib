"""Tests for Sequence and Transition bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    PrimitiveType,
    DebugInfo,
    Sequence,
    Block,
)


class TestSequence:
    """Test suite for Sequence properties."""

    def test_root_is_sequence(self):
        """Test that SDFG root is a Sequence."""
        builder = StructuredSDFGBuilder("test_sdfg")
        sdfg = builder.move()

        root = sdfg.root
        assert isinstance(root, Sequence)

    def test_size_property(self):
        """Test size property returns number of children."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()
        builder.add_block()
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        assert root.size == 3

    def test_child_method(self):
        """Test child() method returns child at index."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        child0 = root.child(0)
        child1 = root.child(1)

        assert isinstance(child0, Block)
        assert isinstance(child1, Block)
        assert child0.element_id != child1.element_id

    def test_at_method(self):
        """Test at() method returns (child, transition) pair."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        child, transition = root.at(0)
        assert isinstance(child, Block)
        assert transition is not None

    def test_transition_method(self):
        """Test transition() method returns transition at index."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        transition = root.transition(0)
        assert transition is not None
        assert transition.element_id >= 0

    def test_children_method(self):
        """Test children() method returns all children."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        children = root.children()
        assert len(children) == 2
        assert all(isinstance(c, Block) for c in children)

    def test_index_method(self):
        """Test index() method finds child index."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        child0 = root.child(0)
        child1 = root.child(1)

        assert root.index(child0) == 0
        assert root.index(child1) == 1

    def test_len_dunder(self):
        """Test __len__ returns number of children."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()
        builder.add_block()
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        assert len(root) == 3

    def test_iter_dunder(self):
        """Test iteration over children using indexing."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        # Use list comprehension with indexing
        children = [root[i] for i in range(len(root))]
        assert len(children) == 2
        assert all(isinstance(c, Block) for c in children)

    def test_sequence_repr(self):
        """Test Sequence string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        repr_str = repr(root)
        assert "Sequence" in repr_str
        assert "children=" in repr_str

    def test_nested_sequence(self):
        """Test nested sequences through control flow constructs."""
        builder = StructuredSDFGBuilder("test_sdfg")

        # Create a for loop which has its own body sequence
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        root = sdfg.root

        # Root has one child (the for loop)
        assert root.size == 1

        # For loop body should be accessible
        for_node = root.child(0)
        from docc.sdfg import For

        assert isinstance(for_node, For)

        # The loop body is a sequence
        body = for_node.body
        assert isinstance(body, Sequence)
        assert body.size == 1


class TestTransition:
    """Test suite for Transition properties."""

    def test_transition_element_id(self):
        """Test that element_id is accessible."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        transition = root.transition(0)
        assert transition.element_id >= 0

    def test_transition_empty(self):
        """Test empty property for transitions without assignments."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        transition = root.transition(0)
        assert transition.empty is True

    def test_transition_size(self):
        """Test size property for transitions."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        transition = root.transition(0)
        assert transition.size == 0

    def test_transition_repr(self):
        """Test Transition string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()

        sdfg = builder.move()
        root = sdfg.root

        transition = root.transition(0)
        repr_str = repr(transition)
        assert "Transition" in repr_str
        assert "assignments=" in repr_str
