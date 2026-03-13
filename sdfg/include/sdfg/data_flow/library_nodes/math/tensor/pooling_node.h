#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Pooling("ml::Pooling");

/**
 * @enum PoolingMode
 * @brief Pooling operation type
 */
enum class PoolingMode {
    Max, ///< Max pooling: output = max(window)
    Sum, ///< Sum pooling: output = sum(window)
    Avg ///< Average pooling: output = sum(window) / window_size
};

/**
 * @class PoolingNode
 * @brief N-dimensional pooling operation
 *
 * PoolingNode represents a pooling operation over spatial dimensions of
 * an NCHW-layout tensor. The operation slides a window over each spatial
 * dimension and reduces each window to a single value using the specified
 * pooling mode (max, sum, or average).
 *
 * ## Input/Output Requirements
 * - Input connector "X": Input tensor [N, C, D1, ..., Dn]
 * - Output connector "Y": Output tensor [N, C, D1_out, ..., Dn_out]
 *
 * ## Supported Modes
 * - Max: Takes the maximum value in each window
 * - Sum: Takes the sum of values in each window
 * - Avg: Takes the average (sum / window_size) of values in each window
 */
class PoolingNode : public TensorNode {
protected:
    PoolingMode mode_;
    std::vector<symbolic::Expression> shape_; ///< Input shape [N, C, D1, ..., Dn]
    std::vector<symbolic::Expression> kernel_shape_; ///< Pooling window shape [k1, ..., kn]
    std::vector<symbolic::Expression> strides_; ///< Stride along each spatial axis
    std::vector<symbolic::Expression> pads_; ///< Padding (start and end for each axis)
    std::vector<symbolic::Expression> dilations_; ///< Dilation along each spatial axis

public:
    PoolingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        PoolingMode mode,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<symbolic::Expression>& kernel_shape,
        const std::vector<symbolic::Expression>& strides,
        const std::vector<symbolic::Expression>& pads,
        const std::vector<symbolic::Expression>& dilations
    );

    PoolingMode mode() const { return mode_; }
    const std::vector<symbolic::Expression>& shape() const { return shape_; }
    const std::vector<symbolic::Expression>& kernel_shape() const { return kernel_shape_; }
    const std::vector<symbolic::Expression>& strides() const { return strides_; }
    const std::vector<symbolic::Expression>& pads() const { return pads_; }
    const std::vector<symbolic::Expression>& dilations() const { return dilations_; }

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    bool supports_integer_types() const override { return true; }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    static std::string mode_to_string(PoolingMode mode);
    static PoolingMode string_to_mode(const std::string& str);
};

/**
 * @class PoolingNodeSerializer
 * @brief Serializer for PoolingNode
 */
class PoolingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
