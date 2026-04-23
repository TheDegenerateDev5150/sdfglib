#include "sdfg/transformations/accumulator_tile.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

/**
 * Test: AccumulatorTile on a 1D accumulator pattern
 *
 * Before:
 *   for i = 0..4:
 *       for j = 0..8:
 *           C[j] += A[j]
 *
 * After AccumulatorTile(i_loop, C[j]):
 *   // init: copy C into tile
 *   for __j = 0..8: C_tile[__j] = C[__j]
 *   // compute with tile
 *   for i = 0..4:
 *       for j = 0..8: C_tile[j] += A[j]
 *   // writeback
 *   for __j = 0..8: C[__j] = C_tile[__j]
 */
TEST(AccumulatorTile, Basic1D) {
    builder::StructuredSDFGBuilder builder("acc_tile_1d", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Array arr_desc(elem_desc, symbolic::integer(8));
    builder.add_container("A", arr_desc, true);
    builder.add_container("C", arr_desc);

    auto& root = builder.subject().root();

    // Outer loop: for i = 0..4
    auto i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    // Inner loop: for j = 0..8
    auto j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // C[j] += A[j]
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {j}, arr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {j}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {j}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::AccumulatorTile transformation(outer_loop, c_in);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);

    // Verify tile buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_accumulator_tile_C"));

    // Structure: root should now contain [init_loop, outer_loop, writeback_loop]
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 3);

    auto* init_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(0).first);
    EXPECT_NE(init_loop, nullptr);

    auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(compute_loop, nullptr);

    auto* wb_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(2).first);
    EXPECT_NE(wb_loop, nullptr);
}

/**
 * Test: AccumulatorTile fails when container is read-only (no writes)
 *
 * for i = 0..4:
 *     for j = 0..8:
 *         B[j] = A[j]   (A is read-only, not an accumulator)
 */
TEST(AccumulatorTile, FailsOnReadOnly) {
    builder::StructuredSDFGBuilder builder("acc_tile_ro", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Array arr_desc(elem_desc, symbolic::integer(8));
    builder.add_container("A", arr_desc, true);
    builder.add_container("B", arr_desc);

    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    auto j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // B[j] = A[j] — A is only read, not accumulated
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {j}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {j}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // A is read-only — should fail
    transformations::AccumulatorTile transformation(outer_loop, a_in);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, am));
}

/**
 * Test: AccumulatorTile fails on scalar (not array/pointer)
 *
 * for i = 0..4:
 *     for j = 0..8:
 *         scalar_val += ...
 */
TEST(AccumulatorTile, FailsOnScalar) {
    builder::StructuredSDFGBuilder builder("acc_tile_scalar", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    builder.add_container("S", elem_desc);

    types::Array arr_desc(elem_desc, symbolic::integer(8));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    auto j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // S += A[j]
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& s_in = builder.add_access(block, "S");
    auto& s_out = builder.add_access(block, "S");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, s_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {j}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", s_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // S is a scalar, not array — should fail
    transformations::AccumulatorTile transformation(outer_loop, s_in);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, am));
}

/**
 * Test: AccumulatorTile fails when there are no inner loops
 *
 * for i = 0..4:
 *     C[i] += 1   (no nested loop — use OutLocalStorage instead)
 */
TEST(AccumulatorTile, FailsWithoutInnerLoop) {
    builder::StructuredSDFGBuilder builder("acc_tile_no_inner", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Array arr_desc(elem_desc, symbolic::integer(4));
    builder.add_container("C", arr_desc);
    builder.add_container("one", elem_desc, true);

    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    // C[i] += one (no inner loop)
    auto& block = builder.add_block(loop.root());
    auto& c_in = builder.add_access(block, "C");
    auto& one_in = builder.add_access(block, "one");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {i}, arr_desc);
    builder.add_computational_memlet(block, one_in, tasklet, "_in2", {}, elem_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {i}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // No inner loops — should fail
    transformations::AccumulatorTile transformation(loop, c_in);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, am));
}

/**
 * Test: AccumulatorTile fails when access node is outside the loop
 *
 * C[0] = ...         (outside)
 * for i = 0..4:
 *     for j = 0..8:
 *         C[j] += A[j]
 */
TEST(AccumulatorTile, FailsOnAccessOutsideLoop) {
    builder::StructuredSDFGBuilder builder("acc_tile_outside", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Array arr_desc(elem_desc, symbolic::integer(8));
    builder.add_container("A", arr_desc, true);
    builder.add_container("C", arr_desc);
    builder.add_container("B", arr_desc);

    auto& root = builder.subject().root();

    // Place an access to C outside the loop
    auto& outer_block = builder.add_block(root);
    auto& b_outside = builder.add_access(outer_block, "B");
    auto& i_outside = builder.add_access(outer_block, "i");
    auto& tasklet_outside = builder.add_tasklet(outer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(outer_block, i_outside, tasklet_outside, "_in", {});
    builder.add_computational_memlet(outer_block, tasklet_outside, "_out", b_outside, {symbolic::integer(0)}, arr_desc);

    auto i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    auto j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // C[j] += A[j] inside the loop
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {j}, arr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {j}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {j}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // c_outside is not inside the loop — should fail because container "C"
    // accesses inside the loop are not associated with this access node
    transformations::AccumulatorTile transformation(outer_loop, b_outside);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, am));
}

/**
 * Test: AccumulatorTile fails when container is write-only (no reads)
 *
 * for i = 0..4:
 *     for j = 0..8:
 *         C[j] = A[j]   (C is only written, not read+written)
 */
TEST(AccumulatorTile, FailsOnWriteOnly) {
    builder::StructuredSDFGBuilder builder("acc_tile_wo", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Array arr_desc(elem_desc, symbolic::integer(8));
    builder.add_container("A", arr_desc, true);
    builder.add_container("C", arr_desc);

    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    auto j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // C[j] = A[j] — write only, no read-modify-write
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {j}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {j}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // C is write-only — should fail
    transformations::AccumulatorTile transformation(outer_loop, c_out);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, am));
}
