#include <gtest/gtest.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "symengine/symengine_rcp.h"

namespace sdfg::rocm {

TEST(ROCMD2HTransferTest, DispatcherNoThrow) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A_host", pointer_type);
    builder.add_container("A_device", pointer_type);
    builder.add_container("N", integer_desc);

    auto [block, d2h_node] = offloading::add_offloading_block<ROCMDataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    ROCMDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), d2h_node);
    EXPECT_NO_THROW(dispatcher.dispatch(pretty_printer, globals_printer, snippet_factory));
}

TEST(ROCMH2DTransferTest, DispatcherNoThrow) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A_host", pointer_type);
    builder.add_container("A_device", pointer_type);
    builder.add_container("N", integer_desc);

    auto [block, h2d_node] = offloading::add_offloading_block<ROCMDataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    ROCMDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), h2d_node);
    EXPECT_NO_THROW(dispatcher.dispatch(pretty_printer, globals_printer, snippet_factory));
}

TEST(ROCMMallocTest, DispatcherNoThrow) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A_device", pointer_type);
    builder.add_container("N", integer_desc);

    auto [block, malloc_node] = offloading::add_offloading_block<ROCMDataOffloadingNode>(
        builder,
        root,
        "A_device",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    ROCMDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), malloc_node);
    EXPECT_NO_THROW(dispatcher.dispatch(pretty_printer, globals_printer, snippet_factory));
}

TEST(ROCMFreeTest, DispatcherNoThrow) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A_device", pointer_type);

    auto [block, free_node] = offloading::add_offloading_block<ROCMDataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        SymEngine::null,
        symbolic::integer(0)
    );

    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    ROCMDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), free_node);
    EXPECT_NO_THROW(dispatcher.dispatch(pretty_printer, globals_printer, snippet_factory));
}

} // namespace sdfg::rocm
