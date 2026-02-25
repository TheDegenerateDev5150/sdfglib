#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/fill_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

FillNode::FillNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Fill, shape) {}

bool FillNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
    const types::Tensor& input_type,
    const types::Tensor& output_type,
    const data_flow::Subset& subset
) {
    // Add code
    auto& code_block = builder.add_block(body);

    auto& input_node_new = builder.add_access(code_block, input_name);
    auto& output_node_new = builder.add_access(code_block, output_name);

    auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in", subset, input_type);
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node_new, subset, output_type);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> FillNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new FillNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

void FillNode::validate(const Function& function) const {
    MathNode::validate(function);

    // Validate that the input is a scalar
    auto& graph = this->get_parent();
    for (auto& iedge : graph.in_edges(*this)) {
        if (iedge.base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "FillNode: Input memlet must be of scalar type. Found type: " + iedge.base_type().print()
            );
        }
    }

    // Check that all output memlets are tensor of scalar
    for (auto& oedge : graph.out_edges(*this)) {
        if (oedge.base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "TensorNode: Output memlet must be of tensor type. Found type: " + oedge.base_type().print()
            );
        }
    }

    // Validate that all memlets have the same primitive type
    types::PrimitiveType prim_type = primitive_type(graph);

    // Check if this operation supports integer types
    if (!supports_integer_types() && types::is_integer(prim_type)) {
        throw InvalidSDFGException(
            "TensorNode: This operation does not support integer types. Found type: " +
            std::string(types::primitive_type_to_string(prim_type))
        );
    }
}

} // namespace tensor
} // namespace math
} // namespace sdfg
