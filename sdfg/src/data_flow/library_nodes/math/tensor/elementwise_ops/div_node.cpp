#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/div_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

DivNode::DivNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseBinaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Div, shape) {}

bool DivNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name_a,
    const std::string& input_name_b,
    const std::string& output_name,
    const types::Tensor& input_type_a,
    const types::Tensor& input_type_b,
    const types::Tensor& output_type,
    const data_flow::Subset& subset
) {
    auto& code_block = builder.add_block(body);

    bool is_int = types::is_integer(output_type.primitive_type());
    data_flow::TaskletCode opcode;
    if (is_int) {
        // Distinguish between signed and unsigned division
        bool is_signed = types::is_signed(output_type.primitive_type());
        opcode = is_signed ? data_flow::TaskletCode::int_sdiv : data_flow::TaskletCode::int_udiv;
    } else {
        opcode = data_flow::TaskletCode::fp_div;
    }
    auto& tasklet = builder.add_tasklet(code_block, opcode, "_out", {"_in1", "_in2"});

    auto& output_node = builder.add_access(code_block, output_name);
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node, subset, output_type);

    create_input_memlet(builder, "_in1", input_name_a, input_type_a, subset, code_block, tasklet);
    create_input_memlet(builder, "_in2", input_name_b, input_type_b, subset, code_block, tasklet);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> DivNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new DivNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
