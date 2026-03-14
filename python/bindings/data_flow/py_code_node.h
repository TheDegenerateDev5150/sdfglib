#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sdfg/data_flow/code_node.h>
#include <sdfg/data_flow/library_node.h>

namespace py = pybind11;

/**
 * @brief Register CodeNode and LibraryNode bindings
 *
 * This function registers the Python bindings for code nodes:
 * - CodeNode: Abstract base class for computational nodes
 * - LibraryNode: Complex operations (BLAS, etc.)
 *
 * Note: Tasklet bindings are already defined in py_tasklet.h/cpp
 */
void register_code_node(py::module& m);
