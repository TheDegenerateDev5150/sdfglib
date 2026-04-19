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
    auto& access = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    auto linearized = symbolic::add(symbolic::mul(i, M), j);
    builder.add_computational_memlet(block, access, tasklet, "in", {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization succeeded
    auto result = analysis.get(memlet);
    ASSERT_NE(result, nullptr);

    // Check subset
    ASSERT_EQ(result->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result->subset.at(1), j));

    // Check layout
    const auto* layout = &result->layout;
    ASSERT_NE(layout, nullptr);
    ASSERT_EQ(layout->shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout->shape().at(0), N));
    EXPECT_TRUE(symbolic::eq(layout->shape().at(1), M));
    ASSERT_EQ(layout->strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout->strides().at(0), M));
    EXPECT_TRUE(symbolic::eq(layout->strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout->offset(), symbolic::zero()));
}

TEST(MemoryLayoutAnalysisTest, DISABLED_Linearized_3D_RowMajor) {
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

    // Define outer loop: for i in [0, N)
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto& i_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // Define inner loop: for j in [0, M)
    auto& j_loop =
        builder.add_for(i_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // Create block with linearized access: A[i*M*K + j*K + k]
    auto& block = builder.add_block(k_loop.root());
    auto& access = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    auto linearized = symbolic::add(symbolic::mul(i, symbolic::mul(M, K)), symbolic::mul(j, K));
    linearized = symbolic::add(linearized, k);
    builder.add_computational_memlet(block, access, tasklet, "in", {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization succeeded
    auto result = analysis.get(memlet);
    ASSERT_NE(result, nullptr);

    // Check subset
    ASSERT_EQ(result->subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(result->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result->subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(result->subset.at(2), k));

    // Check layout
    const auto* layout = &result->layout;
    ASSERT_NE(layout, nullptr);
    ASSERT_EQ(layout->shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout->shape().at(0), N));
    EXPECT_TRUE(symbolic::eq(layout->shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(layout->shape().at(2), K));
    ASSERT_EQ(layout->strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout->strides().at(0), symbolic::mul(M, K)));
    EXPECT_TRUE(symbolic::eq(layout->strides().at(1), K));
    EXPECT_TRUE(symbolic::eq(layout->strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout->offset(), symbolic::zero()));
}

TEST(MemoryLayoutAnalysisTest, DISABLED_Linearized_2D_ColumnMajor) {
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

    auto& inner_loop =
        builder
            .add_for(outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // Create block with column-major linearized access: A[i + j*N]
    auto& block = builder.add_block(inner_loop.root());
    auto& access = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    auto linearized = symbolic::add(i, symbolic::mul(j, N));
    builder.add_computational_memlet(block, access, tasklet, "in", {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization succeeded
    auto result = analysis.get(memlet);
    ASSERT_NE(result, nullptr);

    ASSERT_EQ(result->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result->subset.at(0), j));
    EXPECT_TRUE(symbolic::eq(result->subset.at(1), i));

    const auto* layout = &result->layout;
    ASSERT_NE(layout, nullptr);
    ASSERT_EQ(layout->shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout->shape().at(0), M));
    EXPECT_TRUE(symbolic::eq(layout->shape().at(1), N));
    ASSERT_EQ(layout->strides().size(), 2);
    EXPECT_TRUE(symbolic::eq(layout->strides().at(0), N));
    EXPECT_TRUE(symbolic::eq(layout->strides().at(1), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout->offset(), symbolic::zero()));
}

TEST(MemoryLayoutAnalysisTest, DISABLED_Linearized_3D_ColumnMajor) {
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

    // Loops ordered for column-major traversal: i innermost
    auto& k_loop =
        builder.add_for(root, k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    auto& j_loop =
        builder.add_for(k_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    auto& i_loop =
        builder.add_for(j_loop.root(), i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // Create block with column-major linearized access: A[i + j*N + k*N*M]
    auto& block = builder.add_block(i_loop.root());
    auto& access = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    auto linearized = symbolic::add(i, symbolic::mul(j, N));
    linearized = symbolic::add(linearized, symbolic::mul(k, symbolic::mul(N, M)));
    builder.add_computational_memlet(block, access, tasklet, "in", {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization succeeded
    auto result = analysis.get(memlet);
    ASSERT_NE(result, nullptr);

    ASSERT_EQ(result->subset.size(), 3);
    EXPECT_TRUE(symbolic::eq(result->subset.at(0), k));
    EXPECT_TRUE(symbolic::eq(result->subset.at(1), j));
    EXPECT_TRUE(symbolic::eq(result->subset.at(2), i));

    const auto* layout = &result->layout;
    ASSERT_NE(layout, nullptr);
    ASSERT_EQ(layout->shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout->shape().at(0), K));
    EXPECT_TRUE(symbolic::eq(layout->shape().at(1), M));
    EXPECT_TRUE(symbolic::eq(layout->shape().at(2), N));
    ASSERT_EQ(layout->strides().size(), 3);
    EXPECT_TRUE(symbolic::eq(layout->strides().at(0), symbolic::mul(N, M)));
    EXPECT_TRUE(symbolic::eq(layout->strides().at(1), N));
    EXPECT_TRUE(symbolic::eq(layout->strides().at(2), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(layout->offset(), symbolic::zero()));
}

TEST(MemoryLayoutAnalysisTest, Stencil_2D_5Point) {
    builder::StructuredSDFGBuilder builder("sdfg_stencil", FunctionType_CPU);

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

    // Define outer loop: for i in [1, N-1)
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::sub(N, symbolic::one())), symbolic::one(), symbolic::add(i, symbolic::one())
    );

    // Define inner loop: for j in [1, M-1)
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::sub(M, symbolic::one())),
        symbolic::one(),
        symbolic::add(j, symbolic::one())
    );

    // Create block with 5-point stencil access pattern
    auto& block = builder.add_block(inner_loop.root());

    // Access nodes
    auto& access_A_center = builder.add_access(block, "A");
    auto& access_A_left = builder.add_access(block, "A");
    auto& access_A_right = builder.add_access(block, "A");
    auto& access_A_up = builder.add_access(block, "A");
    auto& access_A_down = builder.add_access(block, "A");
    auto& access_B = builder.add_access(block, "B");

    // Tasklet with 5 inputs for stencil computation
    auto& tasklet =
        builder
            .add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_center", "_left", "_right", "_up", "_down"});

    // Linearized indices for 5-point stencil
    // center: A[i*M + j]
    auto idx_center = symbolic::add(symbolic::mul(i, M), j);
    // left:   A[i*M + (j-1)]
    auto idx_left = symbolic::add(symbolic::mul(i, M), symbolic::sub(j, symbolic::one()));
    // right:  A[i*M + (j+1)]
    auto idx_right = symbolic::add(symbolic::mul(i, M), symbolic::add(j, symbolic::one()));
    // up:     A[(i-1)*M + j]
    auto idx_up = symbolic::add(symbolic::mul(symbolic::sub(i, symbolic::one()), M), j);
    // down:   A[(i+1)*M + j]
    auto idx_down = symbolic::add(symbolic::mul(symbolic::add(i, symbolic::one()), M), j);
    // output: B[i*M + j]
    auto idx_output = symbolic::add(symbolic::mul(i, M), j);

    // Add memlets for reads from A
    auto& memlet_center = builder.add_computational_memlet(block, access_A_center, tasklet, "_center", {idx_center});
    auto& memlet_left = builder.add_computational_memlet(block, access_A_left, tasklet, "_left", {idx_left});
    auto& memlet_right = builder.add_computational_memlet(block, access_A_right, tasklet, "_right", {idx_right});
    auto& memlet_up = builder.add_computational_memlet(block, access_A_up, tasklet, "_up", {idx_up});
    auto& memlet_down = builder.add_computational_memlet(block, access_A_down, tasklet, "_down", {idx_down});
    // Add memlet for write to B
    auto& memlet_output = builder.add_computational_memlet(block, tasklet, "_out", access_B, {idx_output});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Check A memlets - accessed with offsets i±1, j±1, so bounds are N and M
    std::vector<std::pair<std::string, const data_flow::Memlet*>> A_memlets = {
        {"center", &memlet_center},
        {"left", &memlet_left},
        {"right", &memlet_right},
        {"up", &memlet_up},
        {"down", &memlet_down}
    };

    // For A memlets: accessed with offsets i-1, i, i+1 and j-1, j, j+1
    // First dimension: computed from merge using max of bounds → max(N, N-1, N-2)
    // Second dimension: comes from delinearization stride division → M (the actual layout)
    auto expected_A_first_dim =
        symbolic::max(N, symbolic::max(symbolic::sub(N, symbolic::one()), symbolic::sub(N, symbolic::integer(2))));

    for (const auto& [name, memlet] : A_memlets) {
        auto result = analysis.get(*memlet);
        ASSERT_NE(result, nullptr) << "Delinearization failed for memlet: " << name;

        // Check subset dimensionality
        ASSERT_EQ(result->subset.size(), 2) << "Subset size mismatch for memlet: " << name;

        // Check layout shape
        const auto* layout = &result->layout;
        ASSERT_NE(layout, nullptr) << "Layout is null for memlet: " << name;
        ASSERT_EQ(layout->shape().size(), 2) << "Shape dimensions mismatch for memlet: " << name;

        EXPECT_TRUE(symbolic::eq(layout->shape().at(0), expected_A_first_dim))
            << "Shape[0] mismatch for memlet: " << name;
        EXPECT_TRUE(symbolic::eq(layout->shape().at(1), M)) << "Shape[1] != M for memlet: " << name;

        // Check layout strides (row-major: [M, 1])
        ASSERT_EQ(layout->strides().size(), 2) << "Strides dimensions mismatch for memlet: " << name;
        EXPECT_TRUE(symbolic::eq(layout->strides().at(0), M)) << "Stride[0] != M for memlet: " << name;
        EXPECT_TRUE(symbolic::eq(layout->strides().at(1), symbolic::one())) << "Stride[1] != 1 for memlet: " << name;
    }

    // Check B memlet - accessed with i, j only (no offsets)
    // First dimension: tight_upper_bound(i) = N-2, so bound = N-1
    // Second dimension: comes from delinearization → M
    {
        auto result = analysis.get(memlet_output);
        ASSERT_NE(result, nullptr) << "Delinearization failed for output memlet";
        ASSERT_EQ(result->subset.size(), 2) << "Subset size mismatch for output memlet";

        const auto* layout = &result->layout;
        ASSERT_NE(layout, nullptr) << "Layout is null for output memlet";
        ASSERT_EQ(layout->shape().size(), 2) << "Shape dimensions mismatch for output memlet";

        auto expected_B_first_dim = symbolic::sub(N, symbolic::one());
        EXPECT_TRUE(symbolic::eq(layout->shape().at(0), expected_B_first_dim)) << "Shape[0] != N-1 for output memlet";
        EXPECT_TRUE(symbolic::eq(layout->shape().at(1), M)) << "Shape[1] != M for output memlet";

        // Strides should still be row-major
        ASSERT_EQ(layout->strides().size(), 2) << "Strides dimensions mismatch for output memlet";
        EXPECT_TRUE(symbolic::eq(layout->strides().at(0), M)) << "Stride[0] != M for output memlet";
        EXPECT_TRUE(symbolic::eq(layout->strides().at(1), symbolic::one())) << "Stride[1] != 1 for output memlet";
    }

    // Check specific subset values for each memlet
    // center: subset should be [i, j]
    auto result_center = analysis.get(memlet_center);
    EXPECT_TRUE(symbolic::eq(result_center->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_center->subset.at(1), j));

    // left: subset should be [i, j-1]
    auto result_left = analysis.get(memlet_left);
    EXPECT_TRUE(symbolic::eq(result_left->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_left->subset.at(1), symbolic::sub(j, symbolic::one())));

    // right: subset should be [i, j+1]
    auto result_right = analysis.get(memlet_right);
    EXPECT_TRUE(symbolic::eq(result_right->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_right->subset.at(1), symbolic::add(j, symbolic::one())));

    // up: subset should be [i-1, j]
    auto result_up = analysis.get(memlet_up);
    EXPECT_TRUE(symbolic::eq(result_up->subset.at(0), symbolic::sub(i, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(result_up->subset.at(1), j));

    // down: subset should be [i+1, j]
    auto result_down = analysis.get(memlet_down);
    EXPECT_TRUE(symbolic::eq(result_down->subset.at(0), symbolic::add(i, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(result_down->subset.at(1), j));

    // output: subset should be [i, j]
    auto result_output = analysis.get(memlet_output);
    EXPECT_TRUE(symbolic::eq(result_output->subset.at(0), i));
    EXPECT_TRUE(symbolic::eq(result_output->subset.at(1), j));

    // Note: A and B have different accessed ranges due to stencil offsets
    // A is accessed with i±1, j±1 → shape [N, M]
    // B is accessed with i, j only → shape [N-1, M-1]
    // This is expected behavior - we compute the per-container accessed range
}

TEST(MemoryLayoutAnalysisTest, DISABLED_Linearized_2D_Tiled) {
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
    builder.add_container("i_tile", index_type);
    builder.add_container("j", index_type);
    builder.add_container("j_tile", index_type);
    builder.add_container("A", pointer_type, true);

    // Define outer loop: for i in [0, N)
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i = symbolic::symbol("i");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(32)));
    auto& inner_loop = builder.add_for(
        outer_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::integer(32))
    );

    auto& tile_loop_i = builder.add_for(
        inner_loop.root(),
        i_tile,
        symbolic::Lt(i_tile, symbolic::add(i, symbolic::integer(32))),
        i,
        symbolic::add(i_tile, symbolic::one())
    );
    auto& tile_loop_j = builder.add_for(
        tile_loop_i.root(),
        j_tile,
        symbolic::Lt(j_tile, symbolic::add(j, symbolic::integer(32))),
        j,
        symbolic::add(j_tile, symbolic::one())
    );

    // Create block with linearized access: A[i_tile*M + j]
    auto& block = builder.add_block(tile_loop_j.root());
    auto& access = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "", {"in"});
    auto linearized = symbolic::add(symbolic::mul(i_tile, M), j_tile);
    builder.add_computational_memlet(block, access, tasklet, "in", {linearized});

    // Run the analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

    // Get the memlet
    auto& dfg = block.dataflow();
    auto& memlet = *dfg.edges().begin();

    // Check that delinearization succeeded
    auto result = analysis.get(memlet);
    ASSERT_NE(result, nullptr);

    // Check subset
    ASSERT_EQ(result->subset.size(), 2);
    EXPECT_TRUE(symbolic::eq(result->subset.at(0), i_tile));
    EXPECT_TRUE(symbolic::eq(result->subset.at(1), j_tile));

    // Check layout
    const auto* layout = &result->layout;
    ASSERT_NE(layout, nullptr);
    ASSERT_EQ(layout->shape().size(), 2);
    std::cout << "Inferred shape: [" << layout->shape().at(0)->__str__() << ", " << layout->shape().at(1)->__str__()
              << "]\n";
    // TBD
}
