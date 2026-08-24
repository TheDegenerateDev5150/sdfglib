#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop peeling transformation for compound-condition loops
 *
 * This transformation targets a loop whose condition is a conjunction of
 * a canonical (constant-trip) bound and one or more dynamic bounds, e.g.
 *
 *   for (k = k0; k < TK + k0 && k < N; ++k) { body }
 *
 * Instead of splitting into a constant-trip main loop and a variable-trip
 * remainder (which keeps a single shared accumulator addressable and therefore
 * spilled to local memory), this transformation *over-approximates* the loop to
 * its canonical constant-trip bound and moves the dropped dynamic bounds into a
 * predicate guarding the body:
 *
 *   for (k = k0; k < TK + k0; ++k) if (k < N) { body }
 *
 * The trip count is now a compile-time constant, so the compiler can fully
 * unroll the nest and keep register-tile accumulators in registers, while the
 * body predicate preserves the exact set of effectful iterations of the
 * original loop. This works uniformly for parallel (map) and reduction loops and
 * requires no neutral element: out-of-range iterations simply do not execute.
 */
class LoopPeeling : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

public:
    /**
     * @brief Construct a predicated-boundary transformation
     * @param loop The loop with compound conditions to over-approximate + guard
     */
    LoopPeeling(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopPeeling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
