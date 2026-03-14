#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/data_flow_node.h>

namespace py = pybind11;

/**
 * @brief Register DataFlowNode, AccessNode, and ConstantNode bindings
 *
 * This function registers the Python bindings for the data flow node hierarchy:
 * - DataFlowNode: Abstract base class for all dataflow graph nodes
 * - AccessNode: Node representing access to a data container
 * - ConstantNode: Node representing a constant literal value
 */
void register_data_flow_node(py::module& m);
