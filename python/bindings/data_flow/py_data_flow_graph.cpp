#include "py_data_flow_graph.h"

#include <sstream>
#include <vector>

#include <sdfg/data_flow/data_flow_graph.h>

using namespace sdfg::data_flow;

void register_data_flow_graph(py::module& m) {
    py::class_<DataFlowGraph>(m, "DataFlowGraph")
        .def_property_readonly(
            "nodes",
            [](DataFlowGraph& graph) {
                std::vector<DataFlowNode*> result;
                for (auto& node : graph.nodes()) {
                    result.push_back(&node);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all nodes in the dataflow graph"
        )
        .def_property_readonly(
            "edges",
            [](DataFlowGraph& graph) {
                std::vector<Memlet*> result;
                for (auto& edge : graph.edges()) {
                    result.push_back(&edge);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all memlets (edges) in the dataflow graph"
        )
        .def(
            "in_edges",
            [](DataFlowGraph& graph, const DataFlowNode& node) {
                std::vector<const Memlet*> result;
                for (const auto& edge : graph.in_edges(node)) {
                    result.push_back(&edge);
                }
                return result;
            },
            py::arg("node"),
            py::return_value_policy::reference,
            "Get incoming memlets for a node"
        )
        .def(
            "out_edges",
            [](DataFlowGraph& graph, const DataFlowNode& node) {
                std::vector<const Memlet*> result;
                for (const auto& edge : graph.out_edges(node)) {
                    result.push_back(&edge);
                }
                return result;
            },
            py::arg("node"),
            py::return_value_policy::reference,
            "Get outgoing memlets for a node"
        )
        .def("in_degree", &DataFlowGraph::in_degree, py::arg("node"), "Get the number of incoming edges for a node")
        .def("out_degree", &DataFlowGraph::out_degree, py::arg("node"), "Get the number of outgoing edges for a node")
        .def_property_readonly(
            "tasklets",
            [](DataFlowGraph& graph) {
                std::vector<Tasklet*> result;
                for (auto* tasklet : graph.tasklets()) {
                    result.push_back(tasklet);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all tasklets in the dataflow graph"
        )
        .def_property_readonly(
            "library_nodes",
            [](DataFlowGraph& graph) {
                std::vector<LibraryNode*> result;
                for (auto* lib_node : graph.library_nodes()) {
                    result.push_back(lib_node);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all library nodes in the dataflow graph"
        )
        .def_property_readonly(
            "data_nodes",
            [](DataFlowGraph& graph) {
                std::vector<AccessNode*> result;
                for (auto* access_node : graph.data_nodes()) {
                    result.push_back(access_node);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all access nodes in the dataflow graph"
        )
        .def_property_readonly(
            "reads",
            [](const DataFlowGraph& graph) {
                std::vector<const AccessNode*> result;
                for (const auto* access_node : graph.reads()) {
                    result.push_back(access_node);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all read access nodes (sources) in the dataflow graph"
        )
        .def_property_readonly(
            "writes",
            [](const DataFlowGraph& graph) {
                std::vector<const AccessNode*> result;
                for (const auto* access_node : graph.writes()) {
                    result.push_back(access_node);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all write access nodes (sinks) in the dataflow graph"
        )
        .def_property_readonly(
            "sources",
            [](DataFlowGraph& graph) {
                std::vector<DataFlowNode*> result;
                for (auto* node : graph.sources()) {
                    result.push_back(node);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all source nodes (no incoming edges) in the dataflow graph"
        )
        .def_property_readonly(
            "sinks",
            [](DataFlowGraph& graph) {
                std::vector<DataFlowNode*> result;
                for (auto* node : graph.sinks()) {
                    result.push_back(node);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all sink nodes (no outgoing edges) in the dataflow graph"
        )
        .def(
            "predecessors",
            [](const DataFlowGraph& graph, const DataFlowNode& node) {
                std::vector<const DataFlowNode*> result;
                for (const auto* pred : graph.predecessors(node)) {
                    result.push_back(pred);
                }
                return result;
            },
            py::arg("node"),
            py::return_value_policy::reference,
            "Get predecessor nodes"
        )
        .def(
            "successors",
            [](const DataFlowGraph& graph, const DataFlowNode& node) {
                std::vector<const DataFlowNode*> result;
                for (const auto* succ : graph.successors(node)) {
                    result.push_back(succ);
                }
                return result;
            },
            py::arg("node"),
            py::return_value_policy::reference,
            "Get successor nodes"
        )
        .def(
            "topological_sort",
            [](DataFlowGraph& graph) {
                std::vector<DataFlowNode*> result;
                for (auto* node : graph.topological_sort()) {
                    result.push_back(node);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get nodes in topological order"
        )
        .def("__repr__", [](const DataFlowGraph& graph) {
            std::ostringstream oss;
            size_t num_nodes = 0;
            size_t num_edges = 0;
            for (const auto& _ : graph.nodes()) {
                (void) _;
                ++num_nodes;
            }
            for (const auto& _ : graph.edges()) {
                (void) _;
                ++num_edges;
            }
            oss << "<DataFlowGraph nodes=" << num_nodes << " edges=" << num_edges << ">";
            return oss.str();
        });
}
