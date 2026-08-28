#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/atomic_op_node.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/exceptions.h"

using namespace sdfg;
using data_flow::AtomicOpType;
using data_flow::AtomicScalarOpCPUImpl;
using data_flow::AtomicScalarOpCudaImpl;
using data_flow::AtomicScalarOpNode;
using data_flow::AtomicScalarOpRocmImpl;
using types::PrimitiveType;

namespace {

// Helper: create a block containing a single AtomicScalarOpNode.
AtomicScalarOpNode& make_node(
    builder::StructuredSDFGBuilder& builder,
    PrimitiveType data_type,
    AtomicOpType op,
    const data_flow::AtomicScalarOpImpl* impl
) {
    auto& block = builder.add_block(builder.subject().root());
    return static_cast<
        AtomicScalarOpNode&>(builder.add_library_node<AtomicScalarOpNode>(block, DebugInfo(), data_type, op, impl));
}

} // namespace

//
// Implementation singletons & identity
//

TEST(AtomicScalarOpImplTest, SingletonIdentity) {
    // instance() must always return the very same object.
    EXPECT_EQ(AtomicScalarOpCPUImpl::instance(), AtomicScalarOpCPUImpl::instance());
    EXPECT_EQ(AtomicScalarOpCudaImpl::instance(), AtomicScalarOpCudaImpl::instance());
    EXPECT_EQ(AtomicScalarOpRocmImpl::instance(), AtomicScalarOpRocmImpl::instance());

    // Distinct implementations are distinct objects.
    EXPECT_NE(
        static_cast<const data_flow::AtomicScalarOpImpl*>(AtomicScalarOpCPUImpl::instance()),
        static_cast<const data_flow::AtomicScalarOpImpl*>(AtomicScalarOpCudaImpl::instance())
    );
}

TEST(AtomicScalarOpImplTest, TypeNames) {
    EXPECT_EQ(AtomicScalarOpCPUImpl::instance()->type_name(), "CPU");
    EXPECT_EQ(AtomicScalarOpCudaImpl::instance()->type_name(), "CUDA");
    EXPECT_EQ(AtomicScalarOpRocmImpl::instance()->type_name(), "ROCm");
}

TEST(AtomicScalarOpImplTest, ImplementationTypeMatchesTypeName) {
    EXPECT_EQ(AtomicScalarOpCPUImpl::instance()->implementation_type().value(), "CPU");
    EXPECT_EQ(AtomicScalarOpCudaImpl::instance()->implementation_type().value(), "CUDA");
    EXPECT_EQ(AtomicScalarOpRocmImpl::instance()->implementation_type().value(), "ROCm");
}

//
// get_implementation() string mapping
//

TEST(AtomicScalarOpImplTest, GetImplementationByName) {
    EXPECT_EQ(AtomicScalarOpNode::get_implementation("CPU"), AtomicScalarOpCPUImpl::instance());
    EXPECT_EQ(AtomicScalarOpNode::get_implementation("CUDA"), AtomicScalarOpCudaImpl::instance());
    EXPECT_EQ(AtomicScalarOpNode::get_implementation("ROCm"), AtomicScalarOpRocmImpl::instance());
}

TEST(AtomicScalarOpImplTest, GetImplementationInvalidThrows) {
    EXPECT_THROW(AtomicScalarOpNode::get_implementation("Metal"), std::runtime_error);
    EXPECT_THROW(AtomicScalarOpNode::get_implementation(""), std::runtime_error);
}

//
// CPU support matrix
//

TEST(AtomicScalarOpImplTest, CpuSupportsAllOpsForIntegerAndFloat) {
    const auto* cpu = AtomicScalarOpCPUImpl::instance();
    const AtomicOpType ops[] = {AtomicOpType::Add, AtomicOpType::Subtract, AtomicOpType::Min, AtomicOpType::Max};
    const PrimitiveType supported[] = {
        PrimitiveType::Int8,
        PrimitiveType::Int16,
        PrimitiveType::Int32,
        PrimitiveType::Int64,
        PrimitiveType::Int128,
        PrimitiveType::UInt8,
        PrimitiveType::UInt16,
        PrimitiveType::UInt32,
        PrimitiveType::UInt64,
        PrimitiveType::Float,
        PrimitiveType::Double,
        PrimitiveType::Half,
        PrimitiveType::BFloat
    };
    for (auto op : ops) {
        for (auto dt : supported) {
            EXPECT_TRUE(cpu->supports(dt, op)) << "CPU should support " << static_cast<int>(dt);
        }
    }
}

TEST(AtomicScalarOpImplTest, CpuRejectsUnsupportedTypes) {
    const auto* cpu = AtomicScalarOpCPUImpl::instance();
    EXPECT_FALSE(cpu->supports(PrimitiveType::Void, AtomicOpType::Add));
    EXPECT_FALSE(cpu->supports(PrimitiveType::Bool, AtomicOpType::Add));
    EXPECT_FALSE(cpu->supports(PrimitiveType::UInt128, AtomicOpType::Add));
    EXPECT_FALSE(cpu->supports(PrimitiveType::FP128, AtomicOpType::Max));
}

//
// CUDA support matrix
//

TEST(AtomicScalarOpImplTest, CudaAddSupport) {
    const auto* cuda = AtomicScalarOpCudaImpl::instance();
    EXPECT_TRUE(cuda->supports(PrimitiveType::Int32, AtomicOpType::Add));
    EXPECT_TRUE(cuda->supports(PrimitiveType::UInt32, AtomicOpType::Add));
    EXPECT_TRUE(cuda->supports(PrimitiveType::Int64, AtomicOpType::Add));
    EXPECT_TRUE(cuda->supports(PrimitiveType::UInt64, AtomicOpType::Add));
    EXPECT_TRUE(cuda->supports(PrimitiveType::Float, AtomicOpType::Add));
    EXPECT_TRUE(cuda->supports(PrimitiveType::Double, AtomicOpType::Add));
    EXPECT_TRUE(cuda->supports(PrimitiveType::Half, AtomicOpType::Add));
    EXPECT_TRUE(cuda->supports(PrimitiveType::BFloat, AtomicOpType::Add));
    EXPECT_FALSE(cuda->supports(PrimitiveType::Int8, AtomicOpType::Add));
}

TEST(AtomicScalarOpImplTest, CudaSubtractSupport) {
    const auto* cuda = AtomicScalarOpCudaImpl::instance();
    EXPECT_TRUE(cuda->supports(PrimitiveType::Int32, AtomicOpType::Subtract));
    EXPECT_TRUE(cuda->supports(PrimitiveType::UInt32, AtomicOpType::Subtract));
    // No 64-bit or floating point subtract on CUDA.
    EXPECT_FALSE(cuda->supports(PrimitiveType::Int64, AtomicOpType::Subtract));
    EXPECT_FALSE(cuda->supports(PrimitiveType::Float, AtomicOpType::Subtract));
}

TEST(AtomicScalarOpImplTest, CudaMinMaxSupport) {
    const auto* cuda = AtomicScalarOpCudaImpl::instance();
    for (auto op : {AtomicOpType::Min, AtomicOpType::Max}) {
        EXPECT_TRUE(cuda->supports(PrimitiveType::Int32, op));
        EXPECT_TRUE(cuda->supports(PrimitiveType::UInt32, op));
        EXPECT_TRUE(cuda->supports(PrimitiveType::Int64, op));
        EXPECT_TRUE(cuda->supports(PrimitiveType::UInt64, op));
        // CUDA has no floating point min/max atomics.
        EXPECT_FALSE(cuda->supports(PrimitiveType::Float, op));
        EXPECT_FALSE(cuda->supports(PrimitiveType::Double, op));
    }
}

//
// ROCm support matrix
//

TEST(AtomicScalarOpImplTest, RocmSubtractSupport) {
    const auto* rocm = AtomicScalarOpRocmImpl::instance();
    EXPECT_TRUE(rocm->supports(PrimitiveType::Int32, AtomicOpType::Subtract));
    EXPECT_TRUE(rocm->supports(PrimitiveType::UInt32, AtomicOpType::Subtract));
    EXPECT_FALSE(rocm->supports(PrimitiveType::Int64, AtomicOpType::Subtract));
}

TEST(AtomicScalarOpImplTest, RocmMinMaxSupportsFloatingPoint) {
    const auto* rocm = AtomicScalarOpRocmImpl::instance();
    for (auto op : {AtomicOpType::Min, AtomicOpType::Max}) {
        EXPECT_TRUE(rocm->supports(PrimitiveType::Int32, op));
        EXPECT_TRUE(rocm->supports(PrimitiveType::UInt64, op));
        // Unlike CUDA, ROCm supports floating point min/max.
        EXPECT_TRUE(rocm->supports(PrimitiveType::Float, op));
        EXPECT_TRUE(rocm->supports(PrimitiveType::Double, op));
        // But not half precision.
        EXPECT_FALSE(rocm->supports(PrimitiveType::Half, op));
    }
}

// The key divergence between the two GPU backends: floating point min/max.
TEST(AtomicScalarOpImplTest, CudaAndRocmDivergeOnFloatMinMax) {
    const auto* cuda = AtomicScalarOpCudaImpl::instance();
    const auto* rocm = AtomicScalarOpRocmImpl::instance();
    EXPECT_FALSE(cuda->supports(PrimitiveType::Float, AtomicOpType::Min));
    EXPECT_TRUE(rocm->supports(PrimitiveType::Float, AtomicOpType::Min));
}

//
// Node construction & implementation binding
//

TEST(AtomicScalarOpNodeTest, ConstructionStoresProperties) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& node = make_node(builder, PrimitiveType::Float, AtomicOpType::Add, AtomicScalarOpCudaImpl::instance());

    EXPECT_EQ(node.data_type(), PrimitiveType::Float);
    EXPECT_EQ(node.atomic_op(), AtomicOpType::Add);
    EXPECT_EQ(node.implementation_type().value(), "CUDA");
    EXPECT_EQ(node.code(), data_flow::LibraryNodeType_AtomicScalarOp);
    // Node keeps its impl and implementation_type consistent.
    EXPECT_NO_THROW(node.verify_impl_matches());
    EXPECT_NO_THROW(node.verify_impl_exits());
}

TEST(AtomicScalarOpNodeTest, ConstructionRejectsUnsupportedCombination) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    // CUDA has no floating point subtract atomic.
    EXPECT_THROW(
        make_node(builder, PrimitiveType::Float, AtomicOpType::Subtract, AtomicScalarOpCudaImpl::instance()),
        InvalidSDFGException
    );
}

TEST(AtomicScalarOpNodeTest, ValidateSucceedsForSupportedNode) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar base_type(PrimitiveType::Int32);
    types::Pointer ptr(base_type);
    builder.add_container("dst", ptr);
    builder.add_container("src", base_type);

    auto& block = builder.add_block(builder.subject().root());
    auto& dst = builder.add_access(block, "dst");
    auto& src = builder.add_access(block, "src");
    auto& node = builder.add_library_node<AtomicScalarOpNode>(
        block, DebugInfo(), PrimitiveType::Int32, AtomicOpType::Min, AtomicScalarOpRocmImpl::instance()
    );
    builder.add_computational_memlet(block, dst, node, "_dst", {}, ptr);
    builder.add_computational_memlet(block, src, node, "_src", {}, base_type);

    auto sdfg = builder.move();
    EXPECT_NO_THROW(sdfg->validate());
}

//
// Cloning (exercised through deep copy, which invokes AtomicScalarOpNode::clone)
//

TEST(AtomicScalarOpNodeTest, CloneViaDeepCopyPreservesProperties) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& source_node =
        make_node(builder_source, PrimitiveType::Double, AtomicOpType::Add, AtomicScalarOpRocmImpl::instance());

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& root_target = builder_target.subject().root();
    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, builder_source.subject().root());
    deep_copy.copy();

    // Locate the cloned block and its atomic node.
    auto& seq = dynamic_cast<structured_control_flow::Sequence&>(root_target.at(0));
    auto& block = dynamic_cast<structured_control_flow::Block&>(seq.at(0));
    ASSERT_EQ(block.dataflow().nodes().size(), 1);
    auto* cloned = dynamic_cast<AtomicScalarOpNode*>(&(*block.dataflow().nodes().begin()));
    ASSERT_NE(cloned, nullptr);
    EXPECT_NE(cloned, &source_node);

    EXPECT_EQ(cloned->data_type(), PrimitiveType::Double);
    EXPECT_EQ(cloned->atomic_op(), AtomicOpType::Add);
    EXPECT_EQ(cloned->implementation_type().value(), "ROCm");
    EXPECT_NO_THROW(cloned->verify_impl_matches());
    EXPECT_NO_THROW(cloned->verify_impl_exits());
}

//
// Switching the implementation with validation
//

TEST(AtomicScalarOpNodeTest, SwitchImplementationToSupportedUpdatesType) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    // Int32 Min is supported by both CUDA and ROCm.
    auto& node = make_node(builder, PrimitiveType::Int32, AtomicOpType::Min, AtomicScalarOpCudaImpl::instance());
    ASSERT_EQ(node.implementation_type().value(), "CUDA");

    node.switch_implementation(*AtomicScalarOpRocmImpl::instance());

    EXPECT_EQ(node.implementation_type().value(), "ROCm");
    // impl and implementation_type stay in sync.
    EXPECT_NO_THROW(node.verify_impl_matches());
    EXPECT_NO_THROW(node.verify_impl_exits());
}

TEST(AtomicScalarOpNodeTest, SwitchImplementationToUnsupportedThrowsAndKeepsState) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    // Float Min is supported by ROCm but NOT by CUDA.
    auto& node = make_node(builder, PrimitiveType::Float, AtomicOpType::Min, AtomicScalarOpRocmImpl::instance());
    ASSERT_EQ(node.implementation_type().value(), "ROCm");

    EXPECT_THROW(node.switch_implementation(*AtomicScalarOpCudaImpl::instance()), InvalidSDFGException);

    // The failed switch must leave the node untouched.
    EXPECT_EQ(node.implementation_type().value(), "ROCm");
    EXPECT_NO_THROW(node.verify_impl_matches());
    EXPECT_NO_THROW(node.verify_impl_exits());
}
