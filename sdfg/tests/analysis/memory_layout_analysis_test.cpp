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

TEST(MemoryLayoutAnalysisTest, Linearized_3D_RowMajor) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("K", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("k", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    // for i in [0, N)
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    // for j in [0, M)
    auto& middle_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));
    // for k in [0, K)
    auto& inner_loop =
        builder
            .add_for(middle_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // A[i*M*K + j*K + k]
    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::add(symbolic::mul(i, symbolic::mul(M, K)), symbolic::mul(j, K)), k);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(2), k));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(2), K));

    ASSERT_EQ(layout_in.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), symbolic::mul(M, K)));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), K));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout_in.offset(), symbolic::zero()));

    // Check tile at innermost loop (k)
    auto* tile_k = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_k, nullptr);

    ASSERT_EQ(tile_k->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(2), symbolic::zero()));

    ASSERT_EQ(tile_k->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(2), symbolic::sub(K, symbolic::one())));

    auto& tile_k_layout = tile_k->layout;
    ASSERT_EQ(tile_k_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(2), K));

    ASSERT_EQ(tile_k_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(0), symbolic::mul(M, K)));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(1), K));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.offset(), symbolic::zero()));

    // Check tile at middle loop (j)
    auto* tile_j = analysis.tile(middle_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(2), symbolic::zero()));

    ASSERT_EQ(tile_j->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(2), symbolic::sub(K, symbolic::one())));

    auto& tile_j_layout = tile_j->layout;
    ASSERT_EQ(tile_j_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(2), K));

    ASSERT_EQ(tile_j_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(0), symbolic::mul(M, K)));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(1), K));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.offset(), symbolic::zero()));

    // Check tile at outer loop (i)
    auto* tile_i = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(2), symbolic::zero()));

    ASSERT_EQ(tile_i->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), symbolic::sub(N, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(2), symbolic::sub(K, symbolic::one())));

    auto& tile_i_layout = tile_i->layout;
    ASSERT_EQ(tile_i_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(2), K));

    ASSERT_EQ(tile_i_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(0), symbolic::mul(M, K)));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(1), K));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.offset(), symbolic::zero()));
}

TEST(MemoryLayoutAnalysisTest, Linearized_3D_ColMajor) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("K", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("k", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    // for i in [0, N)
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    // for j in [0, M)
    auto& middle_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));
    // for k in [0, K)
    auto& inner_loop =
        builder
            .add_for(middle_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // A[k*N*M + j*N + i]  (column-major)
    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::add(symbolic::mul(k, symbolic::mul(N, M)), symbolic::mul(j, N)), i);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body: delinearization recovers [k, j, i] with strides [N*M, N, 1]
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), k));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(2), i));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(2), N));

    ASSERT_EQ(layout_in.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), symbolic::mul(M, N)));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), N));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout_in.offset(), symbolic::zero()));

    // Check tile at innermost loop (k)
    auto* tile_k = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_k, nullptr);

    ASSERT_EQ(tile_k->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(tile_k->min_subset.at(2), i));

    ASSERT_EQ(tile_k->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(0), symbolic::sub(K, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(tile_k->max_subset.at(2), i));

    auto& tile_k_layout = tile_k->layout;
    ASSERT_EQ(tile_k_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.shape().at(2), N));

    ASSERT_EQ(tile_k_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(0), symbolic::mul(M, N)));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(1), N));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_k_layout.offset(), symbolic::zero()));

    // Check tile at middle loop (j)
    auto* tile_j = analysis.tile(middle_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(2), i));

    ASSERT_EQ(tile_j->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), symbolic::sub(K, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(2), i));

    auto& tile_j_layout = tile_j->layout;
    ASSERT_EQ(tile_j_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.shape().at(2), N));

    ASSERT_EQ(tile_j_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(0), symbolic::mul(M, N)));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(1), N));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_j_layout.offset(), symbolic::zero()));

    // Check tile at outer loop (i)
    auto* tile_i = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(2), symbolic::zero()));

    ASSERT_EQ(tile_i->max_subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), symbolic::sub(K, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), symbolic::sub(M, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(2), symbolic::sub(N, symbolic::one())));

    auto& tile_i_layout = tile_i->layout;
    ASSERT_EQ(tile_i_layout.shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.shape().at(2), N));

    ASSERT_EQ(tile_i_layout.strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(0), symbolic::mul(M, N)));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(1), N));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(tile_i_layout.offset(), symbolic::zero()));
}

TEST(MemoryLayoutAnalysisTest, Stencil_2D_5Point) {
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
    builder.add_container("B", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i in [1, N-1)
    auto& outer_loop = builder.add_for(
        root,
        i,
        symbolic::Lt(i, symbolic::sub(N, symbolic::one())),
        symbolic::integer(1),
        symbolic::add(i, symbolic::one())
    );
    // for j in [1, M-1)
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::sub(M, symbolic::one())),
        symbolic::integer(1),
        symbolic::add(j, symbolic::one())
    );

    // 5-point stencil: B[i*M+j] = A[(i-1)*M+j] + A[(i+1)*M+j] + A[i*M+(j-1)] + A[i*M+(j+1)] + A[i*M+j]
    auto& block = builder.add_block(inner_loop.root());

    auto center = symbolic::add(symbolic::mul(i, M), j);
    auto north = symbolic::add(symbolic::mul(symbolic::sub(i, symbolic::one()), M), j);
    auto south = symbolic::add(symbolic::mul(symbolic::add(i, symbolic::one()), M), j);
    auto west = symbolic::add(symbolic::mul(i, M), symbolic::sub(j, symbolic::one()));
    auto east = symbolic::add(symbolic::mul(i, M), symbolic::add(j, symbolic::one()));

    auto& a_center = builder.add_access(block, "A");
    auto& a_north = builder.add_access(block, "A");
    auto& a_south = builder.add_access(block, "A");
    auto& a_west = builder.add_access(block, "A");
    auto& a_east = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_c", "_n", "_s", "_w", "_e"});

    auto& m_center = builder.add_computational_memlet(block, a_center, tasklet, "_c", {center});
    auto& m_north = builder.add_computational_memlet(block, a_north, tasklet, "_n", {north});
    auto& m_south = builder.add_computational_memlet(block, a_south, tasklet, "_s", {south});
    auto& m_west = builder.add_computational_memlet(block, a_west, tasklet, "_w", {west});
    auto& m_east = builder.add_computational_memlet(block, a_east, tasklet, "_e", {east});
    auto& m_out = builder.add_computational_memlet(block, tasklet, "_out", b_out, {center});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // All 5 input accesses to A should delinearize to 2D with the same layout
    for (auto* memlet : {&m_center, &m_north, &m_south, &m_west, &m_east}) {
        auto result = analysis.access(*memlet);
        ASSERT_NE(result, nullptr);

        ASSERT_EQ(result->subset.size(), 2);
        const auto& layout = result->layout;
        ASSERT_EQ(layout.shape().size(), 2);
        EXPECT_TRUE(symbolic::eq(layout.shape().at(0), symbolic::symbol("__unbounded__")));
        EXPECT_TRUE(symbolic::eq(layout.shape().at(1), M));

        ASSERT_EQ(layout.strides().size(), 2);
        EXPECT_TRUE(symbolic::eq(layout.strides().at(0), M));
        EXPECT_TRUE(symbolic::eq(layout.strides().at(1), symbolic::one()));
    }

    // Check specific delinearized indices
    auto r_center = analysis.access(m_center);
    EXPECT_TRUE(symbolic::eq(r_center->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(r_center->subset.at(1), j));

    auto r_north = analysis.access(m_north);
    EXPECT_TRUE(symbolic::eq(r_north->subset.at(0), symbolic::sub(i, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(r_north->subset.at(1), j));

    auto r_south = analysis.access(m_south);
    EXPECT_TRUE(symbolic::eq(r_south->subset.at(0), symbolic::add(i, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(r_south->subset.at(1), j));

    auto r_west = analysis.access(m_west);
    EXPECT_TRUE(symbolic::eq(r_west->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(r_west->subset.at(1), symbolic::sub(j, symbolic::one())));

    auto r_east = analysis.access(m_east);
    EXPECT_TRUE(symbolic::eq(r_east->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(r_east->subset.at(1), symbolic::add(j, symbolic::one())));

    // Output memlet to B should also delinearize
    auto r_out = analysis.access(m_out);
    ASSERT_NE(r_out, nullptr);
    ASSERT_EQ(r_out->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(r_out->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(r_out->subset.at(1), j));

    // Check tile at inner loop for A: min/max across all 5 stencil points
    // i indices: {i-1, i, i+1}, j indices: {j-1, j, j+1}
    // min/max not simplified: min(i, i-1, i+1) and max(i, i-1, i+1)
    auto* tile_j_A = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_j_A, nullptr);

    auto i_minus_1 = symbolic::sub(i, symbolic::one());
    auto i_plus_1 = symbolic::add(i, symbolic::one());
    auto M_minus_1 = symbolic::sub(M, symbolic::one());
    auto M_minus_2 = symbolic::sub(M, symbolic::integer(2));
    auto M_minus_3 = symbolic::sub(M, symbolic::integer(3));

    ASSERT_EQ(tile_j_A->min_subset.size(), 2);
    // min(i, i-1, i+1) — SymEngine doesn't simplify symbolic min with constant offsets
    EXPECT_TRUE(symbolic::eq(tile_j_A->min_subset.at(0), symbolic::min(symbolic::min(i, i_minus_1), i_plus_1)));
    EXPECT_TRUE(symbolic::eq(tile_j_A->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_j_A->max_subset.size(), 2);
    // max(i, i-1, i+1)
    EXPECT_TRUE(symbolic::eq(tile_j_A->max_subset.at(0), symbolic::max(symbolic::max(i, i_minus_1), i_plus_1)));
    // max(M-1, M-2, M-3) — from j+1 at j=M-2, j at j=M-2, j-1 at j=M-2
    EXPECT_TRUE(symbolic::eq(tile_j_A->max_subset.at(1), symbolic::max(symbolic::max(M_minus_1, M_minus_2), M_minus_3))
    );

    // Check tile at inner loop for B
    auto* tile_j_B = analysis.tile(inner_loop, "B");
    ASSERT_NE(tile_j_B, nullptr);

    ASSERT_EQ(tile_j_B->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_B->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j_B->min_subset.at(1), symbolic::integer(1)));

    ASSERT_EQ(tile_j_B->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j_B->max_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j_B->max_subset.at(1), M_minus_2));

    // Check tile at outer loop for A
    auto* tile_i_A = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i_A, nullptr);

    auto N_minus_1 = symbolic::sub(N, symbolic::one());
    auto N_minus_2 = symbolic::sub(N, symbolic::integer(2));
    auto N_minus_3 = symbolic::sub(N, symbolic::integer(3));

    ASSERT_EQ(tile_i_A->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_A->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i_A->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_i_A->max_subset.size(), 2);
    // max(N-1, N-3, N-2) — from inner tile max i+1 at i=N-2, i-1 at i=N-2, i at i=N-2
    EXPECT_TRUE(symbolic::eq(tile_i_A->max_subset.at(0), symbolic::max(symbolic::max(N_minus_1, N_minus_3), N_minus_2))
    );
    EXPECT_TRUE(symbolic::eq(tile_i_A->max_subset.at(1), symbolic::max(symbolic::max(M_minus_1, M_minus_2), M_minus_3))
    );

    // Check tile at outer loop for B
    auto* tile_i_B = analysis.tile(outer_loop, "B");
    ASSERT_NE(tile_i_B, nullptr);

    ASSERT_EQ(tile_i_B->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_B->min_subset.at(0), symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(tile_i_B->min_subset.at(1), symbolic::integer(1)));

    ASSERT_EQ(tile_i_B->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i_B->max_subset.at(0), N_minus_2));
    EXPECT_TRUE(symbolic::eq(tile_i_B->max_subset.at(1), M_minus_2));
}

TEST(MemoryLayoutAnalysisTest, Linearized_2D_TriangularLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i in [0, N)
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    // for j in [0, i)  — triangular: inner bound depends on outer indvar
    auto& inner_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, i), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // A[i*N + j]
    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(i, N), j);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body: delinearizes to [i, j] with stride N
    auto result_in = analysis.access(memlet_in);
    ASSERT_NE(result_in, nullptr);

    ASSERT_EQ(result_in->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_in->subset.at(1), j));

    const auto& layout_in = result_in->layout;
    ASSERT_EQ(layout_in.shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(0), symbolic::symbol("__unbounded__")));
    EXPECT_TRUE(symbolic::eq(layout_in.shape().at(1), N));

    ASSERT_EQ(layout_in.strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(0), N));
    EXPECT_TRUE(symbolic::eq(layout_in.strides().at(1), symbolic::one()));

    // Check tile at inner loop (j): j ranges [0, i-1], i is constant
    auto* tile_j = analysis.tile(inner_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_j->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), i));
    // j's upper bound is i-1 (from j < i)
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(1), symbolic::sub(i, symbolic::one())));

    // Check tile at outer loop (i): i ranges [0, N-1]
    // Inner tile min=[i, 0], max=[i, i-1]
    // After bounding over i: min=[0, 0], max=[N-1, N-2]
    auto* tile_i = analysis.tile(outer_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), symbolic::zero()));

    ASSERT_EQ(tile_i->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(0), symbolic::sub(N, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(tile_i->max_subset.at(1), symbolic::sub(N, symbolic::integer(2))));
}

TEST(MemoryLayoutAnalysisTest, Linearized_2D_TiledLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar index_type(types::PrimitiveType::Int64);
    types::Scalar scalar_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(scalar_type);
    builder.add_container("N", index_type, true);
    builder.add_container("M", index_type, true);
    builder.add_container("i_tile", index_type);
    builder.add_container("j_tile", index_type);
    builder.add_container("i", index_type);
    builder.add_container("j", index_type);
    builder.add_container("A", pointer_type, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto tile_size = symbolic::integer(32);

    // for i_tile in [0, N) step 32
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, tile_size));

    // for j_tile in [0, M) step 32
    auto& j_tile_loop = builder.add_for(
        i_tile_loop.root(), j_tile, symbolic::Lt(j_tile, M), symbolic::integer(0), symbolic::add(j_tile, tile_size)
    );

    // for i in [i_tile, min(i_tile+32, N)):  cond = i < i_tile+32 && i < N
    auto& i_loop = builder.add_for(
        j_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, tile_size)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // for j in [j_tile, min(j_tile+32, M)):  cond = j < j_tile+32 && j < M
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::And(symbolic::Lt(j, symbolic::add(j_tile, tile_size)), symbolic::Lt(j, M)),
        j_tile,
        symbolic::add(j, symbolic::one())
    );

    // A[i*M + j]
    auto& block = builder.add_block(j_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto linearized = symbolic::add(symbolic::mul(i, M), j);
    auto& memlet_in = builder.add_computational_memlet(block, access_in, tasklet, "_in", {linearized});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {linearized});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check access in body: delinearizes to [i, j] with stride M
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

    // Check tile at j_loop (innermost): j varies, i is constant
    // j ranges [j_tile, min(j_tile+32, M)-1]
    // tight upper bound for j: min(j_tile+31, M-1)
    auto* tile_j = analysis.tile(j_loop, "A");
    ASSERT_NE(tile_j, nullptr);

    ASSERT_EQ(tile_j->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(tile_j->min_subset.at(1), j_tile));

    ASSERT_EQ(tile_j->max_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_j->max_subset.at(0), i));
    // j's tight upper bound: min(j_tile+31, M-1)
    EXPECT_TRUE(symbolic::eq(
        tile_j->max_subset.at(1),
        symbolic::min(symbolic::sub(symbolic::add(j_tile, tile_size), symbolic::one()), symbolic::sub(M, symbolic::one()))
    ));

    // Check tile at i_loop: i varies [i_tile, min(i_tile+32, N)-1], j resolved from inner tile
    auto* tile_i = analysis.tile(i_loop, "A");
    ASSERT_NE(tile_i, nullptr);

    ASSERT_EQ(tile_i->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(0), i_tile));
    EXPECT_TRUE(symbolic::eq(tile_i->min_subset.at(1), j_tile));

    ASSERT_EQ(tile_i->max_subset.size(), 2);
    // i's tight upper bound: min(i_tile+31, N-1)
    EXPECT_TRUE(symbolic::eq(
        tile_i->max_subset.at(0),
        symbolic::min(symbolic::sub(symbolic::add(i_tile, tile_size), symbolic::one()), symbolic::sub(N, symbolic::one()))
    ));
    EXPECT_TRUE(symbolic::eq(
        tile_i->max_subset.at(1),
        symbolic::min(symbolic::sub(symbolic::add(j_tile, tile_size), symbolic::one()), symbolic::sub(M, symbolic::one()))
    ));

    // Check tile at j_tile_loop: j_tile varies [0, M) step 32
    auto* tile_jt = analysis.tile(j_tile_loop, "A");
    ASSERT_NE(tile_jt, nullptr);

    ASSERT_EQ(tile_jt->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_jt->min_subset.at(0), i_tile));
    EXPECT_TRUE(symbolic::eq(tile_jt->min_subset.at(1), symbolic::zero()));

    // Check tile at i_tile_loop (outermost): everything resolved
    auto* tile_it = analysis.tile(i_tile_loop, "A");
    ASSERT_NE(tile_it, nullptr);

    ASSERT_EQ(tile_it->min_subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(tile_it->min_subset.at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(tile_it->min_subset.at(1), symbolic::zero()));
}
