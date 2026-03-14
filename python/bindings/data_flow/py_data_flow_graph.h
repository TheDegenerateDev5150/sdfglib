#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sdfg/data_flow/data_flow_graph.h>

namespace py = pybind11;

/**
 * @brief Register DataFlowGraph bindings
 *
 * This function registers the Python bindings for the DataFlowGraph,
 * which is the container for data flow nodes and memlets.
 */
void register_data_flow_graph(py::module& m);
