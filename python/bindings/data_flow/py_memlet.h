#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sdfg/data_flow/memlet.h>

namespace py = pybind11;

/**
 * @brief Register Memlet bindings
 *
 * This function registers the Python bindings for:
 * - MemletType: Enum for types of data movement
 * - Memlet: Edge in the dataflow graph representing data movement
 */
void register_memlet(py::module& m);
