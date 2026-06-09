#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"

namespace sdfg::cuda::blas {

void generate_kernel_batched_gemm(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const math::blas::BatchedGEMMNode& node
);

class BatchedGEMMNodeDispatcher_CUBLASWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    BatchedGEMMNodeDispatcher_CUBLASWithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::blas::BatchedGEMMNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

class BatchedGEMMNodeDispatcher_CUBLASWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    BatchedGEMMNodeDispatcher_CUBLASWithoutTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::blas::BatchedGEMMNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace sdfg::cuda::blas
