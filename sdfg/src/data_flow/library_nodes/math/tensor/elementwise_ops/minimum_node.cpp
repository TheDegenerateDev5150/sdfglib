#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/minimum_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

MinimumNode::MinimumNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseBinaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Minimum, shape) {}

bool MinimumNode::expand_operation(
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
    auto& output_node = builder.add_access(code_block, output_name);

    bool is_int = types::is_integer(input_type_a.primitive_type());

    data_flow::CodeNode* code_node;
    if (is_int) {
        // Use tasklets for integer types - distinguish between signed and unsigned
        auto tasklet_code = TensorNode::get_integer_minmax_tasklet(input_type_a.primitive_type(), false);
        code_node = &builder.add_tasklet(code_block, tasklet_code, "_out", {"_in1", "_in2"});
    } else {
        // Use intrinsics for floating-point types with correct suffix
        code_node = &builder.add_library_node<
            cmath::CMathNode>(code_block, this->debug_info(), cmath::CMathFunction::fmin, input_type_a.primitive_type());
    }

    create_input_memlet(builder, "_in1", input_name_a, input_type_a, subset, code_block, *code_node);
    create_input_memlet(builder, "_in2", input_name_b, input_type_b, subset, code_block, *code_node);
    builder.add_computational_memlet(code_block, *code_node, "_out", output_node, subset, output_type);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> MinimumNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new MinimumNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
