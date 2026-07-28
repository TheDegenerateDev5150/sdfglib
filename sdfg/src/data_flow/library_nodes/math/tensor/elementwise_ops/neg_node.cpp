#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/neg_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

NegNode::NegNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Neg, shape, "Y", {"X"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput NegNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input = needed_inputs.at(0);

    if (types::is_floating_point(input.required_type)) {
        // Floating-point negation maps directly to a unary fp_neg tasklet.
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_neg, "_out", {"_in"}, this->debug_info());
        input.consumer = &tasklet;
        input.input_conn_index = 0;
        return {.producer = &tasklet, .output_conn_index = 0, .type = input.required_type};
    } else if (types::is_integer(input.required_type)) {
        // Integer negation is expressed as a multiplication with the constant -1.
        auto& const_neg1 = builder.add_constant(block, "-1", types::Scalar(input.required_type), this->debug_info());

        auto& tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::int_mul, "_out", {"_in0", "_in1"}, this->debug_info());
        builder.add_computational_memlet(
            block, const_neg1, tasklet, "_in1", {}, types::Scalar(input.required_type), this->debug_info()
        );
        input.consumer = &tasklet;
        input.input_conn_index = 0;
        return {.producer = &tasklet, .output_conn_index = 0, .type = input.required_type};
    } else {
        throw InvalidSDFGException(
            std::string("NegNode: Unsupported expected type for expand_operation_dataflow: ") +
            types::primitive_type_to_string(input.required_type)
        );
    }
}

std::unique_ptr<data_flow::DataFlowNode> NegNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new NegNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
