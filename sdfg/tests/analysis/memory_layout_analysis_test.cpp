#include "sdfg/analysis/memory_layout_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(MemoryLayoutAnalysisTest, Linearized_2D_RowMajor) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    // Define outer loop: for i in [0, N)
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // Define inner loop: for j in [0, M)
    auto& inner_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // Create block with linearized access: A[i*M + j]
    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(i, M), j);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), j));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), M));

    ASSERT_EQ(layout_in.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), M));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout_in.offset(), symbolic::zero()));

    // Check tile at inner loop
    auto* tile_j = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_j->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), symbolic::sub(M, symbolic::one())));

    auto& tile_j_layout = tile_j->layout;
    ASSERT_EQ(tile_j_layout.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(1), M));

    ASSERT_EQ(tile_j_layout.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(0), M));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.offset(), symbolic::zero()));

    // Check tile at outer loop
    auto* tile_i = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_i->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), symbolic::sub(N, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), symbolic::sub(M, symbolic::one())));

    auto& tile_i_layout = tile_i->layout;
    ASSERT_EQ(tile_i_layout.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(1), M));

    ASSERT_EQ(tile_i_layout.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(0), M));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.offset(), symbolic::zero()));
}

TEST(MemoryLayoutAnalysisTest, Linearized_2D_ColMajor) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    // Define outer loop: for i in [0, N)
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // Define inner loop: for j in [0, M)
    auto& inner_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // Create block with linearized access: A[j*N + i]
    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(j, N), i);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), j));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), i));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), N));

    ASSERT_EQ(layout_in.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), N));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout_in.offset(), symbolic::zero()));

    // Check tile at inner loop
    auto* tile_j = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), i));

    ASSERT_EQ(tile_j->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), i));

    auto& tile_j_layout = tile_j->layout;
    ASSERT_EQ(tile_j_layout.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(1), N));

    ASSERT_EQ(tile_j_layout.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(0), N));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.offset(), symbolic::zero()));

    // Check tile at outer loop
    auto* tile_i = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_i->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), symbolic::sub(N, symbolic::one())));

    auto& tile_i_layout = tile_i->layout;
    ASSERT_EQ(tile_i_layout.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(1), N));

    ASSERT_EQ(tile_i_layout.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(0), N));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.offset(), symbolic::zero()));
}
