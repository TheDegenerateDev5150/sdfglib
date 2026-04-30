#include "sdfg/transformations/out_local_storage.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"

using namespace sdfg;

TEST(OutLocalStorage, Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::integer(4);
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {});

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto& new_root = builder_opt.subject().root();
    // Apply
    transformations::OutLocalStorage transformation(loop, access_out);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Check
    EXPECT_EQ(new_root.size(), 3);
    auto init_block = dynamic_cast<structured_control_flow::Block*>(&new_root.at(0).first);
    EXPECT_NE(init_block, nullptr);
    EXPECT_EQ(init_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(init_block->dataflow().edges().size(), 2);
    bool c_access = false;
    bool a_access = false;
    for (auto& node : init_block->dataflow().nodes()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access->data() == "C") {
                c_access = true;
            } else if (access->data() == "__daisy_out_local_storage_C") {
                a_access = true;
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_TRUE(c_access);

    for (auto& memlet : init_block->dataflow().edges()) {
        if (memlet.dst_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.dst());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "__daisy_out_local_storage_C");
        } else if (memlet.src_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.src());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "C");
        }
    }

    auto new_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(new_loop, nullptr);

    auto& body_loop = new_loop->root();
    EXPECT_EQ(body_loop.size(), 1);
    auto loop_block = dynamic_cast<structured_control_flow::Block*>(&body_loop.at(0).first);
    EXPECT_NE(loop_block, nullptr);
    EXPECT_EQ(loop_block->dataflow().nodes().size(), 4);
    EXPECT_EQ(loop_block->dataflow().edges().size(), 3);
    int accesses = 0;
    a_access = false;
    for (auto access_node : loop_block->dataflow().data_nodes()) {
        if (access_node->data() == "A") {
            a_access = true;
        } else if (access_node->data() == "__daisy_out_local_storage_C") {
            accesses++;
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_EQ(accesses, 2);

    auto deinit_block = dynamic_cast<structured_control_flow::Block*>(&new_root.at(2).first);
    EXPECT_NE(deinit_block, nullptr);

    EXPECT_EQ(deinit_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(deinit_block->dataflow().edges().size(), 2);
    c_access = false;
    a_access = false;
    for (auto& node : deinit_block->dataflow().nodes()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access->data() == "C") {
                c_access = true;
            } else if (access->data() == "__daisy_out_local_storage_C") {
                a_access = true;
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_TRUE(c_access);

    for (auto& memlet : deinit_block->dataflow().edges()) {
        if (memlet.dst_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.dst());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "C");
        } else if (memlet.src_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.src());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "__daisy_out_local_storage_C");
        }
    }
}

TEST(OutLocalStorage, Array) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers — flat pointers with linearized access
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("C", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::integer(100);
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation: C[i] += A[i]
    auto& block = builder.add_block(body);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")}, desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto& new_root = builder_opt.subject().root();
    // Apply
    transformations::OutLocalStorage transformation(loop, access_out);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Check: [init_loop, compute_loop, writeback_loop]
    EXPECT_EQ(new_root.size(), 3);

    // Init loop: copy C to C_local
    auto init_for = dynamic_cast<structured_control_flow::For*>(&new_root.at(0).first);
    EXPECT_NE(init_for, nullptr);
    EXPECT_TRUE(symbolic::eq(init_for->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(init_for->condition(), symbolic::Lt(init_for->indvar(), symbolic::integer(100))));

    auto& init_body = init_for->root();
    EXPECT_EQ(init_body.size(), 1);
    auto init_block = dynamic_cast<structured_control_flow::Block*>(&init_body.at(0).first);
    EXPECT_NE(init_block, nullptr);
    bool c_access = false;
    bool a_access = false;
    for (auto& node : init_block->dataflow().nodes()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access->data() == "C")
                c_access = true;
            else if (access->data() == "__daisy_out_local_storage_C")
                a_access = true;
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_TRUE(c_access);

    // Compute loop preserved
    auto new_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(new_loop, nullptr);

    auto& body_loop = new_loop->root();
    EXPECT_EQ(body_loop.size(), 1);
    auto loop_block = dynamic_cast<structured_control_flow::Block*>(&body_loop.at(0).first);
    EXPECT_NE(loop_block, nullptr);
    int accesses = 0;
    a_access = false;
    for (auto access_node : loop_block->dataflow().data_nodes()) {
        if (access_node->data() == "A")
            a_access = true;
        else if (access_node->data() == "__daisy_out_local_storage_C")
            accesses++;
    }
    EXPECT_TRUE(a_access);
    EXPECT_EQ(accesses, 2);

    // Writeback loop: copy C_local back to C
    auto wb_for = dynamic_cast<structured_control_flow::For*>(&new_root.at(2).first);
    EXPECT_NE(wb_for, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_for->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_for->condition(), symbolic::Lt(wb_for->indvar(), symbolic::integer(100))));

    auto& wb_body = wb_for->root();
    EXPECT_EQ(wb_body.size(), 1);
    auto wb_block = dynamic_cast<structured_control_flow::Block*>(&wb_body.at(0).first);
    EXPECT_NE(wb_block, nullptr);
    c_access = false;
    a_access = false;
    for (auto& node : wb_block->dataflow().nodes()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access->data() == "C")
                c_access = true;
            else if (access->data() == "__daisy_out_local_storage_C")
                a_access = true;
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_TRUE(c_access);
}

TEST(OutLocalStorage, Fail) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Place an access to A outside the loop (before it, in the root sequence).
    auto& outer_block = builder.add_block(root);
    auto& access_outside = builder.add_access(outer_block, "C");
    auto& access_i_outside = builder.add_access(outer_block, "i");
    auto& tasklet_outside = builder.add_tasklet(outer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(outer_block, access_i_outside, tasklet_outside, "_in", {});
    builder.add_computational_memlet(outer_block, tasklet_outside, "_out", access_outside, {symbolic::integer(0)}, desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation inside the loop (only writes to A)
    auto& block = builder.add_block(body);
    auto& access_in = builder.add_access(block, "i");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::integer(0)}, desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply with the outside access node — should fail
    transformations::OutLocalStorage transformation(loop, access_outside);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

/**
 * Test: OutLocalStorage on inner-loop accumulator with flat pointers
 *
 * Before:
 *   for i = 0..4:
 *       for j = 0..8:
 *           C[j] += A[j]
 *
 * After OutLocalStorage(i_loop, C):
 *   for __d0 = 0..8: C_local[__d0] = C[__d0]    // init (read-write)
 *   for i = 0..4:
 *       for j = 0..8: C_local[j] += A[j]         // compute on tile
 *   for __d0 = 0..8: C[__d0] = C_local[__d0]    // writeback
 */
TEST(OutLocalStorage, InnerLoopAccumulator) {
    builder::StructuredSDFGBuilder builder("ols_inner_acc", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

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
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {j}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {j}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {j}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage transformation(outer_loop, c_in);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);

    // Verify local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C"));

    // Structure: root should now contain [init_loop, outer_loop, writeback_loop]
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 3);

    auto* init_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(0).first);
    EXPECT_NE(init_loop, nullptr);
    // Init loop should iterate 0..8
    EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), symbolic::integer(8))));

    auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(compute_loop, nullptr);

    auto* wb_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(2).first);
    EXPECT_NE(wb_loop, nullptr);
    // Writeback loop should iterate 0..8
    EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), symbolic::integer(8))));

    // Check init loop body: C → tasklet(assign) → C_local
    auto& init_body = init_loop->root();
    EXPECT_EQ(init_body.size(), 1);
    auto* init_block = dynamic_cast<structured_control_flow::Block*>(&init_body.at(0).first);
    EXPECT_NE(init_block, nullptr);
    bool has_c = false, has_local = false;
    for (auto* node : init_block->dataflow().data_nodes()) {
        if (node->data() == "C") has_c = true;
        if (node->data() == "__daisy_out_local_storage_C") has_local = true;
    }
    EXPECT_TRUE(has_c);
    EXPECT_TRUE(has_local);

    // Check writeback loop body: C_local → tasklet(assign) → C
    auto& wb_body = wb_loop->root();
    EXPECT_EQ(wb_body.size(), 1);
    auto* wb_block = dynamic_cast<structured_control_flow::Block*>(&wb_body.at(0).first);
    EXPECT_NE(wb_block, nullptr);
    has_c = false;
    has_local = false;
    for (auto* node : wb_block->dataflow().data_nodes()) {
        if (node->data() == "C") has_c = true;
        if (node->data() == "__daisy_out_local_storage_C") has_local = true;
    }
    EXPECT_TRUE(has_c);
    EXPECT_TRUE(has_local);
}

/**
 * Test: OutLocalStorage on write-only inner-loop pattern with flat pointers
 *
 * Before:
 *   for i = 0..4:
 *       for j = 0..8:
 *           C[j] = A[j]   (write-only, no read of C)
 *
 * After OutLocalStorage(i_loop, C):
 *   for i = 0..4:
 *       for j = 0..8: C_local[j] = A[j]          // compute on tile
 *   for __d0 = 0..8: C[__d0] = C_local[__d0]    // writeback only (no init!)
 */
TEST(OutLocalStorage, WriteOnly) {
    builder::StructuredSDFGBuilder builder("ols_write_only", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

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

    // C[j] = A[j] — write only, no read of C
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {j}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {j}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage transformation(outer_loop, c_out);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);

    // Verify local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C"));

    // Structure: root should now contain [outer_loop, writeback_loop] — NO init!
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 2);

    auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(0).first);
    EXPECT_NE(compute_loop, nullptr);

    auto* wb_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(wb_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), symbolic::integer(8))));
}

/**
 * Test: OutLocalStorage fails on read-only container
 *
 * for i = 0..4:
 *     for j = 0..8:
 *         B[j] = A[j]   (A is read-only, not written)
 */
TEST(OutLocalStorage, FailsOnReadOnly) {
    builder::StructuredSDFGBuilder builder("ols_ro_fail", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc);

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

    // B[j] = A[j] — A is only read
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {j}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {j}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // A is read-only — should fail
    transformations::OutLocalStorage transformation(outer_loop, a_in);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, am));
}

/**
 * Test: OutLocalStorage fails on access node outside the loop
 */
TEST(OutLocalStorage, FailsOnAccessOutsideLoop) {
    builder::StructuredSDFGBuilder builder("ols_outside_fail", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);
    builder.add_container("B", ptr_desc);

    auto& root = builder.subject().root();

    // Place an access to B outside the loop
    auto& outer_block = builder.add_block(root);
    auto& b_outside = builder.add_access(outer_block, "B");
    auto& i_outside = builder.add_access(outer_block, "i");
    auto& tasklet_outside = builder.add_tasklet(outer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(outer_block, i_outside, tasklet_outside, "_in", {});
    builder.add_computational_memlet(outer_block, tasklet_outside, "_out", b_outside, {symbolic::integer(0)}, ptr_desc);

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
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {j}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {j}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {j}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // b_outside is not associated with loop body — should fail
    transformations::OutLocalStorage transformation(outer_loop, b_outside);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, am));
}

/**
 * Test: OutLocalStorage with flat pointer and linearized 2D access
 *
 * Before:
 *   for i = 0..4:
 *       for j = 0..8:
 *           C[i*8+j] += A[i*8+j]
 *
 * After OutLocalStorage(i_loop, C):
 *   for d0 = 0..4:
 *       for d1 = 0..8:
 *           C_local[d0*8+d1] = C[d0*8+d1]
 *   for i = 0..4:
 *       for j = 0..8:
 *           C_local[(i-0)*8+(j-0)] += A[i*8+j]
 *   for d0 = 0..4:
 *       for d1 = 0..8:
 *           C[d0*8+d1] = C_local[d0*8+d1]
 *
 * Buffer size: 4 * 8 = 32 (linearized flat pointer)
 */
TEST(OutLocalStorage, FlatPointer_Linearized2D) {
    builder::StructuredSDFGBuilder builder("ols_flat_2d", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // Outer loop: for i = 0..4
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    // Inner loop: for j = 0..8
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // C[i*8+j] += A[i*8+j]
    auto linear_idx = symbolic::add(symbolic::mul(i, symbolic::integer(8)), j);
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {linear_idx}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {linear_idx}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {linear_idx}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage transformation(outer_loop, c_in);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);

    // Verify local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C"));

    // Structure: [init_loop(s), outer_loop, writeback_loop(s)]
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 3);

    // Init should be a for loop (first dimension)
    auto* init_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(0).first);
    EXPECT_NE(init_loop, nullptr);
    // Should iterate 0..4 (first dim extent)
    EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), symbolic::integer(4))));

    // Init loop should contain nested loop for second dimension
    auto& init_body = init_loop->root();
    EXPECT_EQ(init_body.size(), 1);
    auto* inner_init = dynamic_cast<structured_control_flow::For*>(&init_body.at(0).first);
    EXPECT_NE(inner_init, nullptr);
    EXPECT_TRUE(symbolic::eq(inner_init->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(inner_init->condition(), symbolic::Lt(inner_init->indvar(), symbolic::integer(8))));

    // Compute loop preserved
    auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(compute_loop, nullptr);

    // Writeback should be a for loop
    auto* wb_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(2).first);
    EXPECT_NE(wb_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), symbolic::integer(4))));
}

/**
 * Test: OutLocalStorage with tiled loop and symbolic bounds
 *
 * Before:
 *   for i_tile = 0..N step MC:
 *       for k_tile = 0..K step KC:
 *           for i = i_tile..min(i_tile+MC, N):
 *               for k = k_tile..min(k_tile+KC, K):
 *                   C[i*K+k] += A[i*K+k]
 *
 * After OutLocalStorage(i_loop, C):
 *   The tile extents at i_loop level are MC x KC (constant after overapprox):
 *   for i_tile = 0..N step MC:
 *       for k_tile = 0..K step KC:
 *           // Init: copy MC*KC tile from C
 *           for d0 = 0..MC:
 *               for d1 = 0..KC:
 *                   C_local[d0*KC+d1] = C[linearize(i_tile+d0, k_tile+d1)]
 *           // Compute
 *           for i = i_tile..min(i_tile+MC, N):
 *               for k = k_tile..min(k_tile+KC, K):
 *                   C_local[...] += A[i*K+k]
 *           // Writeback
 *           for d0 = 0..MC:
 *               for d1 = 0..KC:
 *                   C[linearize(i_tile+d0, k_tile+d1)] = C_local[d0*KC+d1]
 *
 * Buffer size: MC * KC (constant, known at compile time)
 */
TEST(OutLocalStorage, TiledAccumulator_2D) {
    builder::StructuredSDFGBuilder builder("ols_tiled_2d", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("K", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("k_tile", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("k", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    auto MC = symbolic::integer(64);
    auto KC = symbolic::integer(128);
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    auto i_tile = symbolic::symbol("i_tile");
    auto k_tile = symbolic::symbol("k_tile");
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");

    // for i_tile = 0; i_tile < N; i_tile += MC
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, MC));

    // for k_tile = 0; k_tile < K; k_tile += KC
    auto& k_tile_loop =
        builder
            .add_for(i_tile_loop.root(), k_tile, symbolic::Lt(k_tile, K), symbolic::integer(0), symbolic::add(k_tile, KC));

    // for i = i_tile; i < min(i_tile+MC, N); i++
    auto& i_loop = builder.add_for(
        k_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, MC)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // for k = k_tile; k < min(k_tile+KC, K); k++
    auto& k_loop = builder.add_for(
        i_loop.root(),
        k,
        symbolic::And(symbolic::Lt(k, symbolic::add(k_tile, KC)), symbolic::Lt(k, K)),
        k_tile,
        symbolic::add(k, symbolic::one())
    );

    // C[i*K+k] += A[i*K+k]
    auto linear_idx = symbolic::add(symbolic::mul(i, K), k);
    auto& block = builder.add_block(k_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {linear_idx}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {linear_idx}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {linear_idx}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply OutLocalStorage at i_loop level (tile at this level has extents MC x KC)
    transformations::OutLocalStorage transformation(i_loop, c_in);

    bool can_apply = transformation.can_be_applied(builder_opt, am);
    EXPECT_TRUE(can_apply);

    if (can_apply) {
        transformation.apply(builder_opt, am);

        // Verify local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C"));

        // Structure: k_tile_loop body should be [init_loops, i_loop, writeback_loops]
        auto& k_tile_body = k_tile_loop.root();
        EXPECT_EQ(k_tile_body.size(), 3);

        // Init: nested for loops (MC x KC)
        auto* init_loop = dynamic_cast<structured_control_flow::For*>(&k_tile_body.at(0).first);
        EXPECT_NE(init_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), MC)));

        // Check nested second dimension
        auto& init_inner_body = init_loop->root();
        EXPECT_EQ(init_inner_body.size(), 1);
        auto* init_inner = dynamic_cast<structured_control_flow::For*>(&init_inner_body.at(0).first);
        EXPECT_NE(init_inner, nullptr);
        EXPECT_TRUE(symbolic::eq(init_inner->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(init_inner->condition(), symbolic::Lt(init_inner->indvar(), KC)));

        // Compute loop preserved
        auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&k_tile_body.at(1).first);
        EXPECT_NE(compute_loop, nullptr);

        // Writeback: nested for loops (MC x KC)
        auto* wb_loop = dynamic_cast<structured_control_flow::For*>(&k_tile_body.at(2).first);
        EXPECT_NE(wb_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), MC)));
    }
}

/**
 * Test: OutLocalStorage write-only with tiled loop (no init, only writeback)
 *
 * Before:
 *   for i_tile = 0..N step TILE:
 *       for i = i_tile..min(i_tile+TILE, N):
 *           C[i] = f(A[i])   (write-only to C)
 *
 * After OutLocalStorage(inner_loop, C):
 *   for i_tile = 0..N step TILE:
 *       for i = i_tile..min(i_tile+TILE, N):
 *           C_local[i - i_tile] = f(A[i])
 *       for d0 = 0..TILE:
 *           C[i_tile + d0] = C_local[d0]   // writeback only
 *
 * No init loop because C is write-only within inner_loop scope.
 */
TEST(OutLocalStorage, TiledWriteOnly_1D) {
    builder::StructuredSDFGBuilder builder("ols_tiled_wo", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    auto TILE = symbolic::integer(32);
    auto N = symbolic::symbol("N");
    auto i_tile = symbolic::symbol("i_tile");
    auto i = symbolic::symbol("i");

    // for i_tile = 0; i_tile < N; i_tile += TILE
    auto& tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, TILE));

    // for i = i_tile; i < min(i_tile+TILE, N); i++
    auto& inner_loop = builder.add_for(
        tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, TILE)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // C[i] = A[i]  (write-only to C)
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {i}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {i}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply OutLocalStorage at inner_loop level (tile has extent TILE)
    transformations::OutLocalStorage transformation(inner_loop, c_out);

    bool can_apply = transformation.can_be_applied(builder_opt, am);
    EXPECT_TRUE(can_apply);

    if (can_apply) {
        transformation.apply(builder_opt, am);

        // Verify local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C"));

        // Structure: tile_loop body should be [inner_loop, writeback_loop] — NO init!
        auto& tile_body = tile_loop.root();
        EXPECT_EQ(tile_body.size(), 2);

        // First: compute loop
        auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&tile_body.at(0).first);
        EXPECT_NE(compute_loop, nullptr);

        // Second: writeback loop (0..TILE)
        auto* wb_loop = dynamic_cast<structured_control_flow::For*>(&tile_body.at(1).first);
        EXPECT_NE(wb_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), TILE)));

        // Writeback body: C_local → assign → C
        auto& wb_body = wb_loop->root();
        EXPECT_EQ(wb_body.size(), 1);
        auto* wb_block = dynamic_cast<structured_control_flow::Block*>(&wb_body.at(0).first);
        EXPECT_NE(wb_block, nullptr);
        bool has_c = false, has_local = false;
        for (auto* node : wb_block->dataflow().data_nodes()) {
            if (node->data() == "C") has_c = true;
            if (node->data() == "__daisy_out_local_storage_C") has_local = true;
        }
        EXPECT_TRUE(has_c);
        EXPECT_TRUE(has_local);
    }
}

/**
 * Test: OutLocalStorage with flat pointer linearized 1D access (non-zero base)
 *
 * Before:
 *   for i_tile = 0..N step TILE:
 *       for j = 0..M:
 *           for i = i_tile..min(i_tile+TILE, N):
 *               C[i] += A[j*N+i]
 *
 * After OutLocalStorage(j_loop, C):
 *   The tile of C at j_loop level has extent TILE, base i_tile.
 *   for i_tile = 0..N step TILE:
 *       for d0 = 0..TILE: C_local[d0] = C[i_tile+d0]    // init
 *       for j = 0..M:
 *           for i = i_tile..min(i_tile+TILE, N):
 *               C_local[i-i_tile] += A[j*N+i]
 *       for d0 = 0..TILE: C[i_tile+d0] = C_local[d0]    // writeback
 */
TEST(OutLocalStorage, TiledAccumulator_1D_NonZeroBase) {
    builder::StructuredSDFGBuilder builder("ols_tiled_1d_base", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    auto TILE = symbolic::integer(64);
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i_tile = symbolic::symbol("i_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i_tile = 0; i_tile < N; i_tile += TILE
    auto& tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, TILE));

    // for j = 0; j < M; j++
    auto& j_loop =
        builder
            .add_for(tile_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // for i = i_tile; i < min(i_tile+TILE, N); i++
    auto& i_loop = builder.add_for(
        j_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, TILE)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // C[i] += A[j*N+i]
    auto& block = builder.add_block(i_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {i}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {symbolic::add(symbolic::mul(j, N), i)}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {i}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply OutLocalStorage at j_loop level (tile has extent TILE from inner i_loop)
    transformations::OutLocalStorage transformation(j_loop, c_in);

    bool can_apply = transformation.can_be_applied(builder_opt, am);
    EXPECT_TRUE(can_apply);

    if (can_apply) {
        transformation.apply(builder_opt, am);

        // Verify local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C"));

        // Structure: tile_loop body should be [init_loop, j_loop, writeback_loop]
        auto& tile_body = tile_loop.root();
        EXPECT_EQ(tile_body.size(), 3);

        // Init loop: 0..TILE
        auto* init_loop = dynamic_cast<structured_control_flow::For*>(&tile_body.at(0).first);
        EXPECT_NE(init_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), TILE)));

        // Init body should read from C and write to C_local
        auto& init_body = init_loop->root();
        EXPECT_EQ(init_body.size(), 1);
        auto* init_block = dynamic_cast<structured_control_flow::Block*>(&init_body.at(0).first);
        EXPECT_NE(init_block, nullptr);
        bool has_c = false, has_local = false;
        for (auto* node : init_block->dataflow().data_nodes()) {
            if (node->data() == "C") has_c = true;
            if (node->data() == "__daisy_out_local_storage_C") has_local = true;
        }
        EXPECT_TRUE(has_c);
        EXPECT_TRUE(has_local);

        // Compute loop (j_loop) preserved
        auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&tile_body.at(1).first);
        EXPECT_NE(compute_loop, nullptr);

        // Writeback loop: 0..TILE
        auto* wb_loop = dynamic_cast<structured_control_flow::For*>(&tile_body.at(2).first);
        EXPECT_NE(wb_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), TILE)));
    }
}
