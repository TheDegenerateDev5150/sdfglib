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
    types::Array a_desc(elem_desc, symbolic::integer(4));
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
    transformations::InLocalStorage transformation(loop, "A");
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
    types::Array arr_desc(elem_desc, symbolic::integer(4));
    builder.add_container("A", arr_desc, true); // read-only
    builder.add_container("C", arr_desc, true); // read-write

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
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {indvar}, arr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {indvar}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on C since it's written
    transformations::InLocalStorage ils_c(loop, "C");
    EXPECT_FALSE(ils_c.can_be_applied(builder_opt, am));

    // InLocalStorage should SUCCEED on A since A is read-only
    transformations::InLocalStorage ils_a(loop, "A");
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
    transformations::InLocalStorage ils(loop, "scalar_val");
    EXPECT_FALSE(ils.can_be_applied(builder_opt, am));
}

/**
 * Test: InLocalStorage should fail when container doesn't exist
 */
TEST(InLocalStorage, FailsOnNonexistent) {
    builder::StructuredSDFGBuilder builder("ils_nonexist_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Array arr_desc(elem_desc, symbolic::integer(4));
    builder.add_container("A", arr_desc, true);

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
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on nonexistent container
    transformations::InLocalStorage ils(loop, "B_nonexistent");
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
    types::Array arr_desc(elem_desc, symbolic::integer(4));
    builder.add_container("A", arr_desc, true);
    builder.add_container("B", arr_desc, true); // declared but not used

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // Only use A, not B
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on B (not used in loop body)
    transformations::InLocalStorage ils(loop, "B");
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
    types::Array arr_desc(elem_desc, symbolic::integer(4));
    builder.add_container("A", arr_desc, true);
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
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    analysis::AnalysisManager am(builder.subject());

    // Create transformation and serialize
    transformations::InLocalStorage original(loop, "A");
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
