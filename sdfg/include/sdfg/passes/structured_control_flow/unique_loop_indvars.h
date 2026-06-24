#pragma once

#include <string>
#include <unordered_set>

#include "sdfg/passes/pass.h"

namespace sdfg {

namespace structured_control_flow {
class ControlFlowNode;
}

namespace passes {

/**
 * @brief Ensures every structured loop has a unique induction variable.
 *
 * Traverses all structured loops in the SDFG. Each loop's induction variable is
 * recorded in a set of already-used names. If a loop reuses an induction variable
 * that has already been seen (e.g. two sibling or nested loops both named `i`),
 * the loop's induction variable is renamed to a fresh container and every use of
 * the old symbol within that loop (init, update, condition and body) is replaced.
 *
 * Disambiguating shared induction variables is required for downstream analyses
 * (e.g. argument-size/subset inference for offloading) that key on the symbol of
 * a loop variable and would otherwise treat a nested loop variable as a free symbol.
 */
class UniqueLoopIndvars : public Pass {
private:
    void process_node(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::ControlFlowNode& node,
        std::unordered_set<std::string>& used_indvars,
        bool& applied
    );

public:
    UniqueLoopIndvars();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
