#include "sdfg/analysis/assumptions_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(AssumptionsAnalysisTest, Init_bool) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Bool);
    builder.add_container("N", desc, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).upper_bounds().begin(), symbolic::one()));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 1);
    EXPECT_TRUE(analysis.is_parameter("N"));
}

TEST(AssumptionsAnalysisTest, Init_i8) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt8);
    types::Scalar desc_signed(types::PrimitiveType::Int8);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).upper_bounds().begin(), symbolic::integer(255)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).lower_bounds().begin(), symbolic::integer(-128)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).upper_bounds().begin(), symbolic::integer(127)));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, Init_i16) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt16);
    types::Scalar desc_signed(types::PrimitiveType::Int16);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).upper_bounds().begin(), symbolic::integer(65535)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).lower_bounds().begin(), symbolic::integer(-32768)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).upper_bounds().begin(), symbolic::integer(32767)));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, Init_i32) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt32);
    types::Scalar desc_signed(types::PrimitiveType::Int32);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).upper_bounds().begin(), symbolic::integer(4294967295))
    );
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).lower_bounds().begin(), symbolic::integer(-2147483648))
    );
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("M")).upper_bounds().begin(), symbolic::integer(2147483647))
    );
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, Init_i64) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_signed, true);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(root, true);

    // Check
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*assumptions.at(symbolic::symbol("N")).upper_bounds().begin(), SymEngine::Inf));
    EXPECT_TRUE(symbolic::
                    eq(*assumptions.at(symbolic::symbol("M")).lower_bounds().begin(),
                       symbolic::integer(std::numeric_limits<int64_t>::min())));
    EXPECT_TRUE(symbolic::
                    eq(*assumptions.at(symbolic::symbol("M")).upper_bounds().begin(),
                       symbolic::integer(std::numeric_limits<int64_t>::max())));
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("N")).tight_upper_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_lower_bound().is_null());
    EXPECT_TRUE(assumptions.at(symbolic::symbol("M")).tight_upper_bound().is_null());
    EXPECT_EQ(analysis.parameters().size(), 2);
    EXPECT_TRUE(analysis.is_parameter("N"));
    EXPECT_TRUE(analysis.is_parameter("M"));
}

TEST(AssumptionsAnalysisTest, For_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("i", desc_signed);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop.root());

    // Check
    EXPECT_EQ(assumptions.size(), 2);
    auto& i_assumptions = assumptions.at(symbolic::symbol("i"));
    EXPECT_EQ(i_assumptions.lower_bounds().size(), 1);
    EXPECT_EQ(i_assumptions.upper_bounds().size(), 1);
    EXPECT_TRUE(!i_assumptions.tight_lower_bound().is_null());
    EXPECT_TRUE(!i_assumptions.tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(*i_assumptions.lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_lower_bound(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(*i_assumptions.upper_bounds().begin(), symbolic::sub(symbolic::symbol("N"), symbolic::one()))
    );
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_upper_bound(), symbolic::sub(symbolic::symbol("N"), symbolic::one())));

    auto& n_assumptions = assumptions.at(symbolic::symbol("N"));
    EXPECT_EQ(n_assumptions.lower_bounds().size(), 1);
    EXPECT_EQ(n_assumptions.upper_bounds().size(), 0);
    EXPECT_TRUE(n_assumptions.tight_lower_bound().is_null());
    EXPECT_TRUE(n_assumptions.tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(*n_assumptions.lower_bounds().begin(), symbolic::one()));
}

TEST(AssumptionsAnalysisTest, For_1D_And) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("M", desc_unsigned, true);
    builder.add_container("i", desc_signed);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::And(symbolic::Le(indvar, bound), symbolic::Le(indvar, symbolic::symbol("M")));
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop.root());

    // Check
    EXPECT_EQ(assumptions.size(), 3);
    auto& i_assumptions = assumptions.at(symbolic::symbol("i"));
    EXPECT_EQ(i_assumptions.lower_bounds().size(), 1);
    EXPECT_EQ(i_assumptions.upper_bounds().size(), 2);
    EXPECT_TRUE(!i_assumptions.tight_lower_bound().is_null());
    EXPECT_TRUE(!i_assumptions.tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(*i_assumptions.lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_lower_bound(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::
                    eq(i_assumptions.tight_upper_bound(), symbolic::min(symbolic::symbol("N"), symbolic::symbol("M"))));
    bool found_m = false;
    bool found_n = false;
    for (const auto& ub : i_assumptions.upper_bounds()) {
        if (symbolic::eq(ub, symbolic::symbol("N"))) {
            found_n = true;
        } else if (symbolic::eq(ub, symbolic::symbol("M"))) {
            found_m = true;
        }
    }
    EXPECT_TRUE(found_n);
    EXPECT_TRUE(found_m);
}

TEST(AssumptionsAnalysisTest, For_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc_unsigned(types::PrimitiveType::UInt64);
    types::Scalar desc_signed(types::PrimitiveType::Int64);
    builder.add_container("N", desc_unsigned, true);
    builder.add_container("i", desc_signed);
    builder.add_container("j", desc_signed);

    // Define loop
    auto bound = symbolic::sub(symbolic::symbol("N"), symbolic::one());
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    auto bound_2 = symbolic::symbol("N");
    auto indvar_2 = symbolic::symbol("j");
    auto init_2 = symbolic::add(indvar, symbolic::one());
    auto condition_2 = symbolic::Lt(indvar_2, bound_2);
    auto update_2 = symbolic::add(indvar_2, symbolic::one());

    auto& loop2 = builder.add_for(loop.root(), indvar_2, condition_2, init_2, update_2);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(loop2.root());

    // Check
    EXPECT_EQ(assumptions.size(), 3);
    auto& i_assumptions = assumptions.at(symbolic::symbol("i"));
    EXPECT_EQ(i_assumptions.lower_bounds().size(), 1);
    EXPECT_EQ(i_assumptions.upper_bounds().size(), 1);
    EXPECT_TRUE(!i_assumptions.tight_lower_bound().is_null());
    EXPECT_TRUE(!i_assumptions.tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(*i_assumptions.lower_bounds().begin(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_lower_bound(), symbolic::integer(0)));
    EXPECT_TRUE(
        symbolic::eq(*i_assumptions.upper_bounds().begin(), symbolic::sub(symbolic::symbol("N"), symbolic::integer(2)))
    );
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_upper_bound(), symbolic::sub(symbolic::symbol("N"), symbolic::integer(2)))
    );

    auto& j_assumptions = assumptions.at(symbolic::symbol("j"));
    EXPECT_EQ(j_assumptions.lower_bounds().size(), 1);
    EXPECT_EQ(j_assumptions.upper_bounds().size(), 1);
    EXPECT_TRUE(!j_assumptions.tight_lower_bound().is_null());
    EXPECT_TRUE(!j_assumptions.tight_upper_bound().is_null());
    EXPECT_TRUE(symbolic::eq(*j_assumptions.lower_bounds().begin(), symbolic::add(symbolic::symbol("i"), symbolic::one()))
    );
    EXPECT_TRUE(symbolic::eq(j_assumptions.tight_lower_bound(), symbolic::add(symbolic::symbol("i"), symbolic::one())));
    EXPECT_TRUE(symbolic::eq(*j_assumptions.upper_bounds().begin(), symbolic::sub(symbolic::symbol("N"), symbolic::one()))
    );
    EXPECT_TRUE(symbolic::eq(j_assumptions.tight_upper_bound(), symbolic::sub(symbolic::symbol("N"), symbolic::one())));
}

// Pins the current conservative behavior: AssumptionsAnalysis does NOT
// inject IfElse branch conditions into the body scope. Inside
//   for i in [0, N):
//     if (i >= 3 && i < N - 3): /* body */
// the body scope's assumption for `i` reflects only the surrounding loop
// (`i in [0, N - 1]`), not the tightened range `[3, N - 4]` implied by the
// then-branch's condition.
//
// This is the upstream root cause of the Regression_Im2col_* MLA tests and
// of the matching ArgumentsAnalysis fallback in arguments_analysis.cpp.
//
// TODO: when IfElse-condition propagation lands in
// AssumptionsAnalysis::traverse, flip the expectations below to assert the
// tightened bounds and update the dependent MLA regression tests.
TEST(AssumptionsAnalysisTest, IfElse_BranchConditionNotPropagated) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar n_type(types::PrimitiveType::Int64);
    builder.add_container("N", n_type, true);
    builder.add_container("i", n_type);

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto three = symbolic::integer(3);

    // for i in [0, N): if (i >= 3 && i < N - 3) { /* taken */ }
    auto& loop = builder.add_for(root, i, symbolic::Lt(i, N), symbolic::zero(), symbolic::add(i, symbolic::one()));
    auto& ife = builder.add_if_else(loop.root());
    auto cond = symbolic::And(symbolic::Ge(i, three), symbolic::Lt(i, symbolic::sub(N, three)));
    auto& taken = builder.add_case(ife, cond);

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& assumptions = analysis.get(taken);

    auto& i_assumptions = assumptions.at(i);

    // Loop-derived bounds are present.
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_lower_bound(), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(i_assumptions.tight_upper_bound(), symbolic::sub(N, symbolic::one())));

    // CURRENT BEHAVIOR: branch condition (`i >= 3 && i < N - 3`) is NOT
    // added to `i`'s constraints. None of the bounds equal 3 or N - 3.
    for (const auto& lb : i_assumptions.lower_bounds()) {
        EXPECT_FALSE(symbolic::eq(lb, three))
            << "IfElse branch condition leaked into lower_bounds — flip this test when IfElse propagation lands.";
    }
    for (const auto& ub : i_assumptions.upper_bounds()) {
        EXPECT_FALSE(symbolic::eq(ub, symbolic::sub(N, symbolic::integer(4))))
            << "IfElse branch condition leaked into upper_bounds — flip this test when IfElse propagation lands.";
    }
}
