#include "sdfg/transformations/loop_peeling.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/if_else.h"

using namespace sdfg;

/// Build for(i = M; i < M + 8 && i < N; i++) { A[i] = A[i] } with a compound condition.
static builder::StructuredSDFGBuilder make_compound_loop(structured_control_flow::For*& out_loop) {
    builder::StructuredSDFGBuilder builder("pb_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto init = symbolic::symbol("M");
    auto canonical = symbolic::add(symbolic::symbol("M"), symbolic::integer(8));
    auto dynamic = symbolic::symbol("N");
    auto condition = symbolic::And(symbolic::Lt(indvar, canonical), symbolic::Lt(indvar, dynamic));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar}, desc);

    out_loop = &loop;
    return builder;
}

TEST(LoopPeelingTest, OverApproximatesAndGuardsBody) {
    structured_control_flow::For* orig = nullptr;
    auto builder = make_compound_loop(orig);

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder b(sdfg);
    analysis::AnalysisManager am(b.subject());

    transformations::LoopPeeling t(*orig);
    EXPECT_TRUE(t.can_be_applied(b, am));
    t.apply(b, am);

    auto& s = b.subject();
    ASSERT_EQ(s.root().size(), 1);

    // The loop is over-approximated AND shifted to 0-based with a literal
    // constant trip count (8), so clang can fully unroll it.
    auto* loop = dyn_cast<structured_control_flow::For*>(&s.root().at(0));
    ASSERT_TRUE(loop != nullptr);
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(loop->condition(), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(8))));

    // Body is a single-case IfElse whose guard is the original condition rewritten
    // for the shifted induction variable (i -> i + M). The over-approximated range
    // is a superset, so re-checking it reproduces the original iterations exactly.
    ASSERT_EQ(loop->root().size(), 1);
    auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&loop->root().at(0));
    ASSERT_TRUE(if_else != nullptr);
    ASSERT_EQ(if_else->size(), 1);
    auto case0 = if_else->at(0);
    auto shifted_i = symbolic::add(symbolic::symbol("i"), symbolic::symbol("M"));
    auto original_cond = symbolic::
        And(symbolic::Lt(symbolic::symbol("i"), symbolic::add(symbolic::symbol("M"), symbolic::integer(8))),
            symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")));
    auto expected_guard = symbolic::subs(original_cond, symbolic::symbol("i"), shifted_i);
    EXPECT_TRUE(symbolic::eq(case0.second, expected_guard));

    // The guarded case holds the original computation.
    ASSERT_EQ(case0.first.size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&case0.first.at(0)) != nullptr);
}

TEST(LoopPeelingTest, NotApplicableToSimpleLoop) {
    builder::StructuredSDFGBuilder builder("pb_simple", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto& loop =
        builder.add_for(root, indvar, condition, symbolic::integer(0), symbolic::add(indvar, symbolic::integer(1)));
    builder.add_block(loop.root());

    auto sdfg_moved = builder.move();
    builder::StructuredSDFGBuilder b(sdfg_moved);
    analysis::AnalysisManager am(b.subject());

    transformations::LoopPeeling t(loop);
    EXPECT_FALSE(t.can_be_applied(b, am));
}
