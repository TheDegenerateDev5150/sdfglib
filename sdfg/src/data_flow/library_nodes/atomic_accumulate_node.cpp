#include "sdfg/data_flow/library_nodes/atomic_accumulate_node.h"

#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace data_flow {

AtomicAccumulateNode::AtomicAccumulateNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const ImplementationType& implementation_type
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_AtomicAccumulate,
          {}, // no outputs: the store happens through the `_dst` pointer input
          {"_dst", "_src"},
          true, // atomic store to (global) memory is an observable side effect
          implementation_type
      ) {

      };

void AtomicAccumulateNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet AtomicAccumulateNode::symbols() const { return {}; };

std::unique_ptr<DataFlowNode> AtomicAccumulateNode::
    clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent) const {
    return std::unique_ptr<AtomicAccumulateNode>(
        new AtomicAccumulateNode(element_id, this->debug_info_, vertex, parent, this->implementation_type_)
    );
};

void AtomicAccumulateNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    // The addressing lives on the in memlets, not the node.
};

void AtomicAccumulateNode::replace(const symbolic::ExpressionMapping& replacements) {
    // The addressing lives on the in memlets, not the node.
};

nlohmann::json AtomicAccumulateNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    if (library_node.code() != data_flow::LibraryNodeType_AtomicAccumulate) {
        throw std::runtime_error("Invalid library node code");
    }
    nlohmann::json j;
    j["code"] = std::string(library_node.code().value());
    j["implementation_type"] = std::string(library_node.implementation_type().value());
    return j;
}

data_flow::LibraryNode& AtomicAccumulateNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, sdfg::structured_control_flow::Block& parent
) {
    auto code = j["code"].get<std::string>();
    if (code != data_flow::LibraryNodeType_AtomicAccumulate.value()) {
        throw std::runtime_error("Invalid library node code");
    }
    data_flow::ImplementationType implementation_type{j.at("implementation_type").get<std::string>()};
    return builder.add_library_node<data_flow::AtomicAccumulateNode>(parent, DebugInfo(), implementation_type);
};

AtomicAccumulateNodeDispatcher::AtomicAccumulateNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const AtomicAccumulateNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void AtomicAccumulateNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    // Input order follows the connector order {"_dst", "_src"}.
    const std::string& dst = inputs.at(0).expr;
    const std::string& src = inputs.at(1).expr;

    const auto impl = this->node_.implementation_type().value();
    const bool gpu = impl == data_flow::ImplementationType_AtomicAccumulate_CUDA.value() ||
                     impl == data_flow::ImplementationType_AtomicAccumulate_ROCM.value();
    const bool cpu = impl == data_flow::ImplementationType_AtomicAccumulate_CPU.value();
    if (!gpu && !cpu) {
        throw std::runtime_error("AtomicAccumulateNode: unsupported implementation_type '" + std::string(impl) + "'");
    }

    // `_src` may be a scalar or a (nested) array staged by LocalStorage; walk its
    // extents so the whole tile is accumulated element-wise. Both operands are
    // indexed by the same logical indices — `_dst`'s type carries the (global)
    // destination strides, `_src`'s the tile strides.
    std::vector<symbolic::Expression> extents;
    const types::IType* elem = &inputs.at(1).edge.base_type();
    if (elem->type_id() == types::TypeID::Pointer && static_cast<const types::Pointer&>(*elem).has_pointee_type()) {
        elem = &static_cast<const types::Pointer&>(*elem).pointee_type();
    }
    while (elem->type_id() == types::TypeID::Array) {
        const auto& arr = static_cast<const types::Array&>(*elem);
        extents.push_back(arr.num_elements());
        elem = &arr.element_type();
    }

    std::string index;
    for (size_t d = 0; d < extents.size(); ++d) {
        std::string iv = "__daisy_aa_" + std::to_string(this->node_.element_id()) + "_" + std::to_string(d);
        out.stream << "for (int " << iv << " = 0; " << iv << " < " << this->language_extension_.expression(extents[d])
                   << "; ++" << iv << ") {" << std::endl;
        out.stream.setIndent(out.stream.indent() + 4);
        index += "[" + iv + "]";
    }

    // A scalar `_dst` is already the element address; an indexed element needs `&`.
    const std::string dst_arg = extents.empty() ? dst : ("&(" + dst + index + ")");
    const std::string src_arg = src + index;

    if (gpu) {
        out.stream << "atomicAdd(" << dst_arg << ", " << src_arg << ");" << std::endl;
    } else {
        out.stream << "#pragma omp atomic" << std::endl;
        out.stream << "*(" << dst_arg << ") += " << src_arg << ";" << std::endl;
    }

    for (size_t d = 0; d < extents.size(); ++d) {
        out.stream.setIndent(out.stream.indent() - 4);
        out.stream << "}" << std::endl;
    }
}

} // namespace data_flow
} // namespace sdfg
