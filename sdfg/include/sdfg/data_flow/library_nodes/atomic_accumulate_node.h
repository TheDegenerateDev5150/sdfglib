#pragma once

#include "sdfg/data_flow/library_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {

namespace data_flow {

inline LibraryNodeCode LibraryNodeType_AtomicAccumulate{"AtomicAccumulate"};

// The implementation_type selects the lowering; the semantics are identical:
// atomically add `_src` into the location `_dst` points at.
inline ImplementationType ImplementationType_AtomicAccumulate_CUDA{"CUDA"};
inline ImplementationType ImplementationType_AtomicAccumulate_ROCM{"ROCm"};
inline ImplementationType ImplementationType_AtomicAccumulate_CPU{"CPU"};

/**
 * @brief Atomically accumulates a value into a (global) accumulator slot.
 *
 * Modelled like `Memcpy`/offloading nodes: two inputs, no output, and no subset
 * on either edge.
 *   - `_dst`: a pointer to (global) memory, already offset to the target element
 *             via a reference memlet, so the store site is `*_dst`.
 *   - `_src`: the value to add (a scalar).
 *
 * It lowers to an atomic add (`atomicAdd(_dst, _src)` on GPU). This is the merge
 * primitive a split-K style reduction uses to combine each block's partial into
 * the shared global accumulator. The `implementation_type` picks CUDA / ROCm /
 * CPU; the codegen is otherwise identical.
 */
class AtomicAccumulateNode : public LibraryNode {
public:
    AtomicAccumulateNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const ImplementationType& implementation_type
    );

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;
};

class AtomicAccumulateNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

class AtomicAccumulateNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    AtomicAccumulateNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::AtomicAccumulateNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace data_flow
} // namespace sdfg
