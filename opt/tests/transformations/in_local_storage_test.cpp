#include "sdfg/transformations/in_local_storage.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/utils.h"

using namespace sdfg;

/**
 * Test: InLocalStorage on a scalar array (1D) with constant bound
 *
 * Before:
 *   for i = 0..4: C += A[i]
 *
 * After:
 *   for i' = 0..4: A_local[i'] = A[i']
 *   for i = 0..4: C += A_local[i]
 */
TEST(InLocalStorage, Scalar_ConstantBound) {
    builder::StructuredSDFGBuilder builder("ils_scalar_test", FunctionType_CPU);

    // Create containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer a_desc(elem_desc);
    builder.add_container("A", a_desc, true); // read-only input
    builder.add_container("C", elem_desc); // accumulator

    auto& root = builder.subject().root();

    // Create loop: for i = 0..4
    auto indvar = symbolic::symbol("i");
    auto bound = symbolic::integer(4);
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation: C += A[i]
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply transformation
    transformations::InLocalStorage transformation(loop, a_in);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);

    // Verify: local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A"));

    // Verify: structure should now be [copy_loop, main_loop]
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 2);

    // First element should be copy loop
    auto* copy_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(0).first);
    EXPECT_NE(copy_loop, nullptr);

    // Second element should be the main loop
    auto* main_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(main_loop, nullptr);

    // Verify copy loop reads from A, writes to A_local
    auto& copy_body = copy_loop->root();
    EXPECT_EQ(copy_body.size(), 1);
    auto* copy_block = dynamic_cast<structured_control_flow::Block*>(&copy_body.at(0).first);
    EXPECT_NE(copy_block, nullptr);

    bool reads_A = false;
    bool writes_A_local = false;
    for (auto* node : copy_block->dataflow().data_nodes()) {
        if (node->data() == "A") reads_A = true;
        if (node->data() == "__daisy_in_local_storage_A") writes_A_local = true;
    }
    EXPECT_TRUE(reads_A);
    EXPECT_TRUE(writes_A_local);

    // Verify main loop uses local buffer
    auto& main_body = main_loop->root();
    EXPECT_EQ(main_body.size(), 1);
    auto* main_block = dynamic_cast<structured_control_flow::Block*>(&main_body.at(0).first);
    EXPECT_NE(main_block, nullptr);

    bool uses_A_local = false;
    bool uses_A_original = false;
    for (auto* node : main_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_in_local_storage_A") uses_A_local = true;
        if (node->data() == "A") uses_A_original = true;
    }
    EXPECT_TRUE(uses_A_local);
    EXPECT_FALSE(uses_A_original); // Original A should be replaced
}

/**
 * Test: InLocalStorage should fail on containers that are written
 *
 * for i = 0..N: C[i] += A[i]
 *
 * InLocalStorage(loop, "C") should fail because C is written
 * InLocalStorage(loop, "A") should succeed because A is read-only
 */
TEST(InLocalStorage, FailsOnWrittenContainer) {
    builder::StructuredSDFGBuilder builder("ils_rw_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true); // read-only
    builder.add_container("C", ptr_desc, true); // read-write

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // C[i] += A[i]
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {indvar}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {indvar}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on C since it's written
    transformations::InLocalStorage ils_c(loop, c_in);
    EXPECT_FALSE(ils_c.can_be_applied(builder_opt, am));

    // InLocalStorage should SUCCEED on A since A is read-only
    transformations::InLocalStorage ils_a(loop, a_in);
    EXPECT_TRUE(ils_a.can_be_applied(builder_opt, am));
}

/**
 * Test: InLocalStorage should fail on scalars
 *
 * for i = 0..N: ... scalar_val ...
 *
 * InLocalStorage(loop, "scalar_val") should fail because it's not an array
 */
TEST(InLocalStorage, FailsOnScalar) {
    builder::StructuredSDFGBuilder builder("ils_scalar_fail_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    builder.add_container("scalar_val", elem_desc, true);
    builder.add_container("result", elem_desc);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // result = result + scalar_val
    auto& block = builder.add_block(loop.root());
    auto& s_in = builder.add_access(block, "scalar_val");
    auto& r_in = builder.add_access(block, "result");
    auto& r_out = builder.add_access(block, "result");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, r_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, s_in, tasklet, "_in2", {}, elem_desc);
    builder.add_computational_memlet(block, tasklet, "_out", r_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on scalar_val
    transformations::InLocalStorage ils(loop, s_in);
    EXPECT_FALSE(ils.can_be_applied(builder_opt, am));
}

/**
 * Test: InLocalStorage should fail when access node is outside the loop
 */
TEST(InLocalStorage, FailsOnAccessOutsideLoop) {
    builder::StructuredSDFGBuilder builder("ils_outside_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc, true);

    auto& root = builder.subject().root();

    // Place an access to B outside the loop
    auto& outer_block = builder.add_block(root);
    auto& b_outside = builder.add_access(outer_block, "B");
    auto& i_outside = builder.add_access(outer_block, "i");
    auto& tasklet_outside = builder.add_tasklet(outer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(outer_block, i_outside, tasklet_outside, "_in", {});
    builder.add_computational_memlet(outer_block, tasklet_outside, "_out", b_outside, {symbolic::integer(0)}, ptr_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on B (not used inside the loop)
    transformations::InLocalStorage ils(loop, b_outside);
    EXPECT_FALSE(ils.can_be_applied(builder_opt, am));
}

/**
 * Test: InLocalStorage should fail when container is not used
 */
TEST(InLocalStorage, FailsOnUnusedContainer) {
    builder::StructuredSDFGBuilder builder("ils_unused_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc, true); // declared but not used

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // Place an access to B outside the loop
    auto& outer_block = builder.add_block(root);
    auto& b_outside = builder.add_access(outer_block, "B");
    auto& i_outside = builder.add_access(outer_block, "i");
    auto& tasklet_outside = builder.add_tasklet(outer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(outer_block, i_outside, tasklet_outside, "_in", {});
    builder.add_computational_memlet(outer_block, tasklet_outside, "_out", b_outside, {symbolic::integer(0)}, ptr_desc);

    // Only use A, not B inside the loop
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on B (not used in loop body)
    transformations::InLocalStorage ils(loop, b_outside);
    EXPECT_FALSE(ils.can_be_applied(builder_opt, am));
}

/**
 * Test: JSON serialization round-trip
 */
TEST(InLocalStorage, JsonSerialization) {
    builder::StructuredSDFGBuilder builder("ils_json_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer a_desc(elem_desc);
    builder.add_container("A", a_desc, true);
    builder.add_container("C", elem_desc);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    analysis::AnalysisManager am(builder.subject());

    // Create transformation and serialize
    transformations::InLocalStorage original(loop, a_in);
    EXPECT_TRUE(original.can_be_applied(builder, am));

    nlohmann::json j;
    original.to_json(j);

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "InLocalStorage");
    EXPECT_EQ(j["container"], "A");
    EXPECT_TRUE(j.contains("subgraph"));

    // Deserialize and verify
    auto deserialized = transformations::InLocalStorage::from_json(builder, j);
    EXPECT_EQ(deserialized.name(), "InLocalStorage");
    EXPECT_TRUE(deserialized.can_be_applied(builder, am));
}

/**
 * Test: InLocalStorage on 2D tiled access (post-tiling of a symbolic loop nest)
 *
 * Before (after tiling i by MC=64, j by NC=32):
 *   for i_tile = 0..M step MC:
 *       for j_tile = 0..N step NC:
 *           for i = i_tile..min(i_tile+MC, M):
 *               for j = j_tile..min(j_tile+NC, N):
 *                   C += A[i*N+j]
 *
 * After InLocalStorage(i_loop, "A"):
 *   Tile at i_loop level has extents MC x NC (integer after overapprox).
 *   for i_tile = 0..M step MC:
 *       for j_tile = 0..N step NC:
 *           for d0 = 0..MC: for d1 = 0..NC:
 *               A_local[d0*NC+d1] = A[(i_tile+d0)*N+(j_tile+d1)]
 *           for i = i_tile..min(i_tile+MC, M):
 *               for j = j_tile..min(j_tile+NC, N):
 *                   C += A_local[(i-i_tile)*NC+(j-j_tile)]
 *
 * Buffer size: MC * NC (constant, known at compile time)
 */
TEST(InLocalStorage, TiledAccess_2D) {
    builder::StructuredSDFGBuilder builder("ils_2d_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("j_tile", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("C", elem_desc);

    types::Pointer a_desc(elem_desc);
    builder.add_container("A", a_desc, true);

    auto& root = builder.subject().root();

    auto MC = symbolic::integer(64);
    auto NC = symbolic::integer(32);
    auto M = symbolic::symbol("M");
    auto N = symbolic::symbol("N");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i_tile = 0; i_tile < M; i_tile += MC
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, M), symbolic::integer(0), symbolic::add(i_tile, MC));

    // for j_tile = 0; j_tile < N; j_tile += NC
    auto& j_tile_loop =
        builder
            .add_for(i_tile_loop.root(), j_tile, symbolic::Lt(j_tile, N), symbolic::integer(0), symbolic::add(j_tile, NC));

    // for i = i_tile; i < min(i_tile+MC, M); i++
    auto& i_loop = builder.add_for(
        j_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, MC)), symbolic::Lt(i, M)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // for j = j_tile; j < min(j_tile+NC, N); j++
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::And(symbolic::Lt(j, symbolic::add(j_tile, NC)), symbolic::Lt(j, N)),
        j_tile,
        symbolic::add(j, symbolic::one())
    );

    // C += A[i*N+j]
    auto& block = builder.add_block(j_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {symbolic::add(symbolic::mul(i, N), j)}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply InLocalStorage at i_loop level (tile has extents MC x NC)
    transformations::InLocalStorage ils(i_loop, a_in);

    bool can_apply = ils.can_be_applied(builder_opt, am);
    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A"));

        // Verify: j_tile_loop body should be [copy_loop, i_loop]
        auto& j_tile_body = j_tile_loop.root();
        EXPECT_EQ(j_tile_body.size(), 2u);

        // First: copy loop (outer dim, 0..MC)
        auto* copy_loop = dynamic_cast<structured_control_flow::For*>(&j_tile_body.at(0).first);
        EXPECT_NE(copy_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(copy_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(copy_loop->condition(), symbolic::Lt(copy_loop->indvar(), MC)));

        // Second: compute loop (i_loop preserved)
        auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&j_tile_body.at(1).first);
        EXPECT_NE(compute_loop, nullptr);
    }
}

/**
 * Test: InLocalStorage with 1D tiled access (post-tiling scenario)
 *
 * Before (after tiling i by TILE=64):
 *   for i_tile = 0..N step TILE:
 *       for i = i_tile..min(i_tile+TILE, N):
 *           C += A[i]
 *
 * After InLocalStorage(inner_loop, "A"):
 *   Tile at inner_loop level has extent TILE (integer after overapprox).
 *   for i_tile = 0..N step TILE:
 *       for d0 = 0..TILE:
 *           A_local[d0] = A[i_tile + d0]
 *       for i = i_tile..min(i_tile+TILE, N):
 *           C += A_local[i - i_tile]
 */
TEST(InLocalStorage, TiledAccess_1D) {
    builder::StructuredSDFGBuilder builder("ils_tiled_1d_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("C", elem_desc);

    // A is a 1D pointer: double* A (size N)
    types::Pointer a_desc(elem_desc);
    builder.add_container("A", a_desc, true);

    auto& root = builder.subject().root();

    // Tile size
    auto TILE = symbolic::integer(64);
    auto N = symbolic::symbol("N");
    auto i_tile = symbolic::symbol("i_tile");
    auto i = symbolic::symbol("i");

    // Outer loop: for i_tile = 0; i_tile < N; i_tile += TILE
    auto& tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, TILE));

    // Inner loop: for i = i_tile; i < min(i_tile+TILE, N); i++
    auto& inner_loop = builder.add_for(
        tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, TILE)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // C += A[i]
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {i}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply InLocalStorage at inner_loop level (tile has extent TILE)
    transformations::InLocalStorage ils(inner_loop, a_in);

    bool can_apply = ils.can_be_applied(builder_opt, am);

    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A"));

        // Verify: tile_loop body should be [copy_loop, inner_loop]
        EXPECT_EQ(tile_loop.root().size(), 2u);

        // First: copy loop (0..TILE)
        auto* copy_loop = dynamic_cast<structured_control_flow::For*>(&tile_loop.root().at(0).first);
        EXPECT_NE(copy_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(copy_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(copy_loop->condition(), symbolic::Lt(copy_loop->indvar(), TILE)));
    }
}

/**
 * Test: InLocalStorage with 2D tiled access (BLIS-style panel packing)
 *
 * Before (after tiling i by MC=64, k by KC=64):
 *   for i_tile = 0..M step MC:
 *       for k_tile = 0..K step KC:
 *           for i = i_tile..min(i_tile+MC, M):
 *               for k = k_tile..min(k_tile+KC, K):
 *                   ... = A[i*K+k]
 *
 * After InLocalStorage(i_loop, "A"):
 *   Tile at i_loop level has extents MC x KC (integer after overapprox).
 *   for i_tile = 0..M step MC:
 *       for k_tile = 0..K step KC:
 *           for d0 = 0..MC: for d1 = 0..KC:
 *               A_local[d0*KC+d1] = A[(i_tile+d0)*K+(k_tile+d1)]
 *           for i = i_tile..min(i_tile+MC, M):
 *               for k = k_tile..min(k_tile+KC, K):
 *                   ... = A_local[(i-i_tile)*KC+(k-k_tile)]
 *
 * Buffer size: MC * KC (constant, known at compile time)
 */
TEST(InLocalStorage, TiledAccess_2D_Panel) {
    builder::StructuredSDFGBuilder builder("ils_tiled_2d_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("M", sym_desc, true);
    builder.add_container("K", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("k_tile", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("k", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("C", elem_desc);

    // A is a 2D array: double A[M][K] (linearized as A[i*K+k])
    types::Pointer a_desc(elem_desc);
    builder.add_container("A", a_desc, true);

    auto& root = builder.subject().root();

    // Tile sizes
    auto MC = symbolic::integer(64);
    auto KC = symbolic::integer(64);
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    auto i_tile = symbolic::symbol("i_tile");
    auto k_tile = symbolic::symbol("k_tile");
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");

    // Outer: for i_tile = 0; i_tile < M; i_tile += MC
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, M), symbolic::integer(0), symbolic::add(i_tile, MC));

    // Next: for k_tile = 0; k_tile < K; k_tile += KC
    auto& k_tile_loop =
        builder
            .add_for(i_tile_loop.root(), k_tile, symbolic::Lt(k_tile, K), symbolic::integer(0), symbolic::add(k_tile, KC));

    // Inner: for i = i_tile; i < i_tile + MC; i++
    auto& i_loop = builder.add_for(
        k_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, MC)), symbolic::Lt(i, M)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // Innermost: for k = k_tile; k < k_tile + KC; k++
    auto& k_loop = builder.add_for(
        i_loop.root(),
        k,
        symbolic::And(symbolic::Lt(k, symbolic::add(k_tile, KC)), symbolic::Lt(k, K)),
        k_tile,
        symbolic::add(k, symbolic::one())
    );

    // C += A[i][k]
    auto& block = builder.add_block(k_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {symbolic::add(symbolic::mul(i, K), k)}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply InLocalStorage at i_loop level (tile has extents MC x KC)
    transformations::InLocalStorage ils(i_loop, a_in);

    bool can_apply = ils.can_be_applied(builder_opt, am);

    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A"));

        // Verify: k_tile_loop body should be [copy_loop, i_loop]
        EXPECT_EQ(k_tile_loop.root().size(), 2u);

        // First: copy loop (outer dim, 0..MC)
        auto* copy_loop = dynamic_cast<structured_control_flow::For*>(&k_tile_loop.root().at(0).first);
        EXPECT_NE(copy_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(copy_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(copy_loop->condition(), symbolic::Lt(copy_loop->indvar(), MC)));

        // Check nested second dimension (0..KC)
        auto& copy_inner_body = copy_loop->root();
        EXPECT_EQ(copy_inner_body.size(), 1u);
        auto* copy_inner = dynamic_cast<structured_control_flow::For*>(&copy_inner_body.at(0).first);
        EXPECT_NE(copy_inner, nullptr);
        EXPECT_TRUE(symbolic::eq(copy_inner->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(copy_inner->condition(), symbolic::Lt(copy_inner->indvar(), KC)));
    }
}

/**
 * Test: InLocalStorage with 2D tiled stencil access (halo region)
 *
 * A 5-point stencil on A[N][M] (linearized as A[i*M+j]) with rectangular tiling.
 *
 * Before (after tiling i by IT=32, j by JT=32):
 *   for i_tile = 1..N-1 step IT:
 *       for j_tile = 1..M-1 step JT:
 *           for i = i_tile..min(i_tile+IT, N-1):
 *               for j = j_tile..min(j_tile+JT, M-1):
 *                   B[i*M+j] = A[(i-1)*M+j] + A[(i+1)*M+j]
 *                            + A[i*M+(j-1)] + A[i*M+(j+1)] + A[i*M+j]
 *
 * After InLocalStorage(i_loop, "A"):
 *   Tile at i_loop level has extents (IT+2) x (JT+2) due to stencil halo.
 *   The stencil accesses i-1..i+IT (overapproximate of i range plus +/-1 halo)
 *   and j-1..j+JT, giving integer extents.
 */
TEST(InLocalStorage, TiledStencil_2D_5Point) {
    builder::StructuredSDFGBuilder builder("ils_tiled_stencil_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("j_tile", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc, true);

    auto& root = builder.subject().root();

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto IT = symbolic::integer(32);
    auto JT = symbolic::integer(32);

    auto N_minus_1 = symbolic::sub(N, symbolic::one());
    auto M_minus_1 = symbolic::sub(M, symbolic::one());

    // for i_tile = 1; i_tile < N-1; i_tile += IT
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N_minus_1), symbolic::one(), symbolic::add(i_tile, IT));

    // for j_tile = 1; j_tile < M-1; j_tile += JT
    auto& j_tile_loop = builder.add_for(
        i_tile_loop.root(), j_tile, symbolic::Lt(j_tile, M_minus_1), symbolic::one(), symbolic::add(j_tile, JT)
    );

    // for i = i_tile; i < min(i_tile+IT, N-1); i++
    auto& i_loop = builder.add_for(
        j_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, IT)), symbolic::Lt(i, N_minus_1)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // for j = j_tile; j < min(j_tile+JT, M-1); j++
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::And(symbolic::Lt(j, symbolic::add(j_tile, JT)), symbolic::Lt(j, M_minus_1)),
        j_tile,
        symbolic::add(j, symbolic::one())
    );

    // 5-point stencil: B[i*M+j] = A[(i-1)*M+j] + A[(i+1)*M+j] + A[i*M+(j-1)] + A[i*M+(j+1)] + A[i*M+j]
    // Each stencil point in its own block with fp_add accumulation into B
    auto& block = builder.add_block(j_loop.root());

    auto center = symbolic::add(symbolic::mul(i, M), j);
    auto north = symbolic::add(symbolic::mul(symbolic::sub(i, symbolic::one()), M), j);
    auto south = symbolic::add(symbolic::mul(symbolic::add(i, symbolic::one()), M), j);
    auto west = symbolic::add(symbolic::mul(i, M), symbolic::sub(j, symbolic::one()));
    auto east = symbolic::add(symbolic::mul(i, M), symbolic::add(j, symbolic::one()));

    // B[center] = A[center]
    auto& a_center = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");
    auto& t0 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_center, t0, "_in", {center});
    builder.add_computational_memlet(block, t0, "_out", b_out, {center});

    // B[center] += A[north]
    auto& block_n = builder.add_block(j_loop.root());
    auto& a_north = builder.add_access(block_n, "A");
    auto& b_in_n = builder.add_access(block_n, "B");
    auto& b_out_n = builder.add_access(block_n, "B");
    auto& t_n = builder.add_tasklet(block_n, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_n, b_in_n, t_n, "_in1", {center});
    builder.add_computational_memlet(block_n, a_north, t_n, "_in2", {north});
    builder.add_computational_memlet(block_n, t_n, "_out", b_out_n, {center});

    // B[center] += A[south]
    auto& block_s = builder.add_block(j_loop.root());
    auto& a_south = builder.add_access(block_s, "A");
    auto& b_in_s = builder.add_access(block_s, "B");
    auto& b_out_s = builder.add_access(block_s, "B");
    auto& t_s = builder.add_tasklet(block_s, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_s, b_in_s, t_s, "_in1", {center});
    builder.add_computational_memlet(block_s, a_south, t_s, "_in2", {south});
    builder.add_computational_memlet(block_s, t_s, "_out", b_out_s, {center});

    // B[center] += A[west]
    auto& block_w = builder.add_block(j_loop.root());
    auto& a_west = builder.add_access(block_w, "A");
    auto& b_in_w = builder.add_access(block_w, "B");
    auto& b_out_w = builder.add_access(block_w, "B");
    auto& t_w = builder.add_tasklet(block_w, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_w, b_in_w, t_w, "_in1", {center});
    builder.add_computational_memlet(block_w, a_west, t_w, "_in2", {west});
    builder.add_computational_memlet(block_w, t_w, "_out", b_out_w, {center});

    // B[center] += A[east]
    auto& block_e = builder.add_block(j_loop.root());
    auto& a_east = builder.add_access(block_e, "A");
    auto& b_in_e = builder.add_access(block_e, "B");
    auto& b_out_e = builder.add_access(block_e, "B");
    auto& t_e = builder.add_tasklet(block_e, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_e, b_in_e, t_e, "_in1", {center});
    builder.add_computational_memlet(block_e, a_east, t_e, "_in2", {east});
    builder.add_computational_memlet(block_e, t_e, "_out", b_out_e, {center});

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply InLocalStorage at i_loop level for A (tile has extents (IT+2) x (JT+2))
    transformations::InLocalStorage ils(i_loop, a_center);

    bool can_apply = ils.can_be_applied(builder_opt, am);

    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A"));

        // Verify: j_tile_loop body should be [copy_loop, i_loop]
        EXPECT_EQ(j_tile_loop.root().size(), 2u);

        // First: copy loop for outer dim (0..(IT+2))
        auto* copy_loop = dynamic_cast<structured_control_flow::For*>(&j_tile_loop.root().at(0).first);
        EXPECT_NE(copy_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(copy_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(copy_loop->condition(), symbolic::Lt(copy_loop->indvar(), symbolic::integer(34))));
    }
}
