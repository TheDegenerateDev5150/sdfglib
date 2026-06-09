#include "sdfg/targets/cuda/blas/batched_gemm.h"
#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"
#include "sdfg/targets/cuda/blas/utils.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg::cuda::blas {

BatchedGEMMNodeDispatcher_CUBLASWithTransfers::BatchedGEMMNodeDispatcher_CUBLASWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::BatchedGEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void BatchedGEMMNodeDispatcher_CUBLASWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const math::blas::BatchedGEMMNode&>(this->node_);

    library_snippet_factory.add_global("#include <cuda.h>");
    library_snippet_factory.add_global("#include <cublas_v2.h>");

    std::string type;
    switch (node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            type = "float";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "double";
            break;
        default:
            throw std::runtime_error("Invalid precision for CUBLAS batched GEMM node");
    }

    auto batch_count_expr = this->language_extension_.expression(node.batch_count());

    std::string size_A = "(" +
                         this->language_extension_.expression(symbolic::mul(node.batch_count(), node.stride_a())) +
                         ") * sizeof(" + type + ")";

    std::string size_B = "(" +
                         this->language_extension_.expression(symbolic::mul(node.batch_count(), node.stride_b())) +
                         ") * sizeof(" + type + ")";

    std::string size_C = "(" +
                         this->language_extension_.expression(symbolic::mul(node.batch_count(), node.stride_c())) +
                         ") * sizeof(" + type + ")";

    // Guard clause
    stream << "if (" << this->language_extension_.expression(node.m()) << " != 0 && "
           << this->language_extension_.expression(node.n()) << " != 0 && "
           << this->language_extension_.expression(node.k()) << " != 0 && " << batch_count_expr << " != 0) {"
           << std::endl;
    stream.setIndent(stream.indent() + 4);

    stream << "cudaError_t err_cuda;" << std::endl;

    stream << type << " *dA, *dB, *dC;" << std::endl;

    stream << "err_cuda = cudaMalloc((void**) &dA, " << size_A << ");" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");
    stream << "err_cuda = cudaMalloc((void**) &dB, " << size_B << ");" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");
    stream << "err_cuda = cudaMalloc((void**) &dC, " << size_C << ");" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");

    stream << "err_cuda = cudaMemcpy(dA, __A, " << size_A << ", cudaMemcpyHostToDevice);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");
    stream << "err_cuda = cudaMemcpy(dB, __B, " << size_B << ", cudaMemcpyHostToDevice);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");
    stream << "err_cuda = cudaMemcpy(dC, __C, " << size_C << ", cudaMemcpyHostToDevice);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");

    setup_blas_handle(library_snippet_factory, this->language_extension_);

    generate_kernel_batched_gemm(stream, this->language_extension_, node);

    stream << "err_cuda = cudaMemcpy(__C, dC, " << size_C << ", cudaMemcpyDeviceToHost);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");

    stream << "err_cuda = cudaFree(dA);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");
    stream << "err_cuda = cudaFree(dB);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");
    stream << "err_cuda = cudaFree(dC);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

BatchedGEMMNodeDispatcher_CUBLASWithoutTransfers::BatchedGEMMNodeDispatcher_CUBLASWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::BatchedGEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void BatchedGEMMNodeDispatcher_CUBLASWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const math::blas::BatchedGEMMNode&>(this->node_);

    library_snippet_factory.add_global("#include <cuda.h>");
    library_snippet_factory.add_global("#include <cublas_v2.h>");

    auto batch_count_expr = this->language_extension_.expression(node.batch_count());

    // Guard clause
    stream << "if (" << this->language_extension_.expression(node.m()) << " != 0 && "
           << this->language_extension_.expression(node.n()) << " != 0 && "
           << this->language_extension_.expression(node.k()) << " != 0 && " << batch_count_expr << " != 0) {"
           << std::endl;
    stream.setIndent(stream.indent() + 4);

    setup_blas_handle(library_snippet_factory, this->language_extension_);

    generate_kernel_batched_gemm(stream, this->language_extension_, node);

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

void generate_kernel_batched_gemm(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const math::blas::BatchedGEMMNode& node
) {
    std::string type;
    switch (node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            type = "S";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "D";
            break;
        default:
            throw std::runtime_error("Invalid precision for CUBLAS batched GEMM node");
    }

    // cuBLAS is column-major native, so for row-major we swap A and B
    auto first_dim = node.m();
    auto second_dim = node.n();
    auto first_mat = "A";
    auto second_mat = "B";
    auto ld_first = node.lda();
    auto ld_second = node.ldb();
    auto ldc = node.ldc();
    auto stride_first = node.stride_a();
    auto stride_second = node.stride_b();
    auto stride_c = node.stride_c();
    auto trans_first = node.trans_a();
    auto trans_second = node.trans_b();

    if (node.layout() == sdfg::math::blas::BLAS_Layout::RowMajor) {
        first_dim = node.n();
        second_dim = node.m();
        first_mat = "B";
        second_mat = "A";
        ldc = node.n();
        stride_first = node.stride_b();
        stride_second = node.stride_a();
        trans_first = node.trans_b();
        trans_second = node.trans_a();
        ld_first = (trans_first == sdfg::math::blas::BLAS_Transpose::No) ? node.n() : node.k();
        ld_second = (trans_second == sdfg::math::blas::BLAS_Transpose::No) ? node.k() : node.m();
    }

    std::string trans_first_str = (trans_first == sdfg::math::blas::BLAS_Transpose::No) ? "CUBLAS_OP_N" : "CUBLAS_OP_T";
    std::string trans_second_str = (trans_second == sdfg::math::blas::BLAS_Transpose::No) ? "CUBLAS_OP_N"
                                                                                          : "CUBLAS_OP_T";

    std::string prefix = node.implementation_type() == cuda::ImplementationType_CUDAWithTransfers ? "d" : "__";

    stream << "cublasStatus_t err;" << std::endl;
    stream << "err = cublas" << type << "gemmStridedBatched(handle, " << trans_first_str << ", " << trans_second_str
           << ", " << language_extension.expression(first_dim) << ", " << language_extension.expression(second_dim)
           << ", " << language_extension.expression(node.k()) << ", "
           << "&__alpha, " << prefix << first_mat << ", " << language_extension.expression(ld_first) << ", "
           << language_extension.expression(stride_first) << ", " << prefix << second_mat << ", "
           << language_extension.expression(ld_second) << ", " << language_extension.expression(stride_second) << ", "
           << "&__beta, " << prefix << "C, " << language_extension.expression(ldc) << ", "
           << language_extension.expression(stride_c) << ", " << language_extension.expression(node.batch_count())
           << ");" << std::endl;
    cublas_error_checking(stream, language_extension, "err");
    check_cuda_kernel_launch_errors(stream, language_extension, false);
}

} // namespace sdfg::cuda::blas
