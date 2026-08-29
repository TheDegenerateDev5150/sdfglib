#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/targets/rocm/rocm.h"

using namespace sdfg;

TEST(RocBlasTest, DotNodeWithDataTransfers) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    auto n = symbolic::integer(10);
    auto stride_a = symbolic::integer(2);
    auto stride_b = symbolic::integer(2);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Array array_desc(desc, n);

    builder.add_container("a", array_desc);
    builder.add_container("b", array_desc);
    builder.add_container("c", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    auto& dot_node = static_cast<math::blas::DotNode&>(builder.add_library_node<math::blas::DotNode>(
        block,
        DebugInfo(),
        rocm::ImplementationType_ROCMWithTransfers,
        math::blas::BLAS_Precision::d,
        n,
        stride_a,
        stride_b
    ));

    builder.add_computational_memlet(block, a_node, dot_node, "__x", {symbolic::zero()}, array_desc, block.debug_info());
    builder.add_computational_memlet(block, b_node, dot_node, "__y", {symbolic::zero()}, array_desc, block.debug_info());
    builder.add_computational_memlet(block, dot_node, "__out", c_node, {}, desc, block.debug_info());

    EXPECT_EQ(block.dataflow().nodes().size(), 4);

    auto outcome = passes::expansion::expand_single_math_node(builder, block, dot_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
}

TEST(RocBlasTest, DotNodeWithoutDataTransfers) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    auto n = symbolic::integer(10);
    auto stride_a = symbolic::integer(2);
    auto stride_b = symbolic::integer(2);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Array array_desc(desc, n);

    builder.add_container("a", array_desc);
    builder.add_container("b", array_desc);
    builder.add_container("c", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    auto& dot_node = static_cast<math::blas::DotNode&>(builder.add_library_node<math::blas::DotNode>(
        block,
        DebugInfo(),
        rocm::ImplementationType_ROCMWithoutTransfers,
        math::blas::BLAS_Precision::d,
        n,
        stride_a,
        stride_b
    ));

    builder.add_computational_memlet(block, a_node, dot_node, "__x", {symbolic::zero()}, array_desc, block.debug_info());
    builder.add_computational_memlet(block, b_node, dot_node, "__y", {symbolic::zero()}, array_desc, block.debug_info());
    builder.add_computational_memlet(block, dot_node, "__out", c_node, {}, desc, block.debug_info());

    EXPECT_EQ(block.dataflow().nodes().size(), 4);

    auto outcome = passes::expansion::expand_single_math_node(builder, block, dot_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
}

TEST(RocBlasTest, GemmNodeWithDataTransfers) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    // res: ixj, A: ixk, B: kxj

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_i)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_k)));
    types::Array arr_res_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_i)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        rocm::ImplementationType_ROCMWithTransfers,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j), // lda
        symbolic::integer(dim_k), // ldb
        symbolic::integer(dim_j) // ldc
    ));

    auto& alpha_node = builder.add_constant(block, "2.0", desc);
    auto& beta_node = builder.add_constant(block, "1.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    EXPECT_EQ(block.dataflow().nodes().size(), 6);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
    builder.subject().validate();

    EXPECT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dyn_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0));
    EXPECT_NE(new_sequence, nullptr);
    // beta == 1: the init nest is skipped, leaving only the compute nest.
    ASSERT_EQ(new_sequence->size(), 1);

    auto comp_map_i = dyn_cast<structured_control_flow::Map*>(&new_sequence->at(0));
    ASSERT_NE(comp_map_i, nullptr);
    EXPECT_EQ(comp_map_i->root().size(), 1);

    auto comp_map_j = dyn_cast<structured_control_flow::Map*>(&comp_map_i->root().at(0));
    ASSERT_NE(comp_map_j, nullptr);
    EXPECT_EQ(comp_map_j->root().size(), 1);

    auto comp_for_k = dyn_cast<structured_control_flow::For*>(&comp_map_j->root().at(0));
    ASSERT_NE(comp_for_k, nullptr);
    EXPECT_EQ(comp_for_k->root().size(), 1);

    auto block_fma = dyn_cast<structured_control_flow::Block*>(&comp_for_k->root().at(0));
    ASSERT_NE(block_fma, nullptr);
    // alpha != 1: p = A[i,k] * B[k,j]; C[i,j] = alpha * p + C[i,j]
    // (a, b, c_in, c_out, fma, mul, prod, alpha)
    EXPECT_EQ(block_fma->dataflow().nodes().size(), 8);

    data_flow::Tasklet* fma_tasklet = nullptr;
    for (auto* tasklet : block_fma->dataflow().tasklets()) {
        if (tasklet->code() == data_flow::TaskletCode::fp_fma) {
            fma_tasklet = tasklet;
        }
    }
    ASSERT_NE(fma_tasklet, nullptr);
    EXPECT_EQ(fma_tasklet->inputs().size(), 3);
    EXPECT_EQ(fma_tasklet->inputs().at(0), "_in1");
    EXPECT_EQ(fma_tasklet->inputs().at(1), "_in2");
    EXPECT_EQ(fma_tasklet->inputs().at(2), "_in3");
    EXPECT_EQ(fma_tasklet->output(), "_out");

    // The accumulating store writes back into C.
    auto& final_edge = *block_fma->dataflow().out_edges(*fma_tasklet).begin();
    auto* final_access = dynamic_cast<data_flow::AccessNode*>(&final_edge.dst());
    EXPECT_NE(final_access, nullptr);
    EXPECT_EQ(final_access->data(), c_var_name);
}

TEST(RocBlasTest, GemmNodeWithoutDataTransfers) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    // res: ixj, A: ixk, B: kxj

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_i)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_k)));
    types::Array arr_res_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_i)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto c_var_name = "output";
    auto& dummy_input_node = builder.add_access(block, c_var_name);
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        rocm::ImplementationType_ROCMWithoutTransfers,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j), // lda
        symbolic::integer(dim_k), // ldb
        symbolic::integer(dim_j) // ldc
    ));

    auto& alpha_node = builder.add_constant(block, "2.0", desc);
    auto& beta_node = builder.add_constant(block, "1.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    EXPECT_EQ(block.dataflow().nodes().size(), 6);

    builder.subject().validate();
    auto outcome = passes::expansion::expand_single_math_node(builder, block, gemm_node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
    builder.subject().validate();

    EXPECT_EQ(sdfg.root().size(), 1);
    auto new_sequence = dyn_cast<structured_control_flow::Sequence*>(&sdfg.root().at(0));
    EXPECT_NE(new_sequence, nullptr);
    // beta == 1: the init nest is skipped, leaving only the compute nest.
    ASSERT_EQ(new_sequence->size(), 1);

    auto comp_map_i = dyn_cast<structured_control_flow::Map*>(&new_sequence->at(0));
    ASSERT_NE(comp_map_i, nullptr);
    EXPECT_EQ(comp_map_i->root().size(), 1);

    auto comp_map_j = dyn_cast<structured_control_flow::Map*>(&comp_map_i->root().at(0));
    ASSERT_NE(comp_map_j, nullptr);
    EXPECT_EQ(comp_map_j->root().size(), 1);

    auto comp_for_k = dyn_cast<structured_control_flow::For*>(&comp_map_j->root().at(0));
    ASSERT_NE(comp_for_k, nullptr);
    EXPECT_EQ(comp_for_k->root().size(), 1);

    auto block_fma = dyn_cast<structured_control_flow::Block*>(&comp_for_k->root().at(0));
    ASSERT_NE(block_fma, nullptr);
    // alpha != 1: p = A[i,k] * B[k,j]; C[i,j] = alpha * p + C[i,j]
    // (a, b, c_in, c_out, fma, mul, prod, alpha)
    EXPECT_EQ(block_fma->dataflow().nodes().size(), 8);

    data_flow::Tasklet* fma_tasklet = nullptr;
    for (auto* tasklet : block_fma->dataflow().tasklets()) {
        if (tasklet->code() == data_flow::TaskletCode::fp_fma) {
            fma_tasklet = tasklet;
        }
    }
    ASSERT_NE(fma_tasklet, nullptr);
    EXPECT_EQ(fma_tasklet->inputs().size(), 3);
    EXPECT_EQ(fma_tasklet->inputs().at(0), "_in1");
    EXPECT_EQ(fma_tasklet->inputs().at(1), "_in2");
    EXPECT_EQ(fma_tasklet->inputs().at(2), "_in3");
    EXPECT_EQ(fma_tasklet->output(), "_out");

    // The accumulating store writes back into C.
    auto& final_edge = *block_fma->dataflow().out_edges(*fma_tasklet).begin();
    auto* final_access = dynamic_cast<data_flow::AccessNode*>(&final_edge.dst());
    EXPECT_NE(final_access, nullptr);
    EXPECT_EQ(final_access->data(), c_var_name);
}
