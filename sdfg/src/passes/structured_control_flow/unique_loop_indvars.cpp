#include "sdfg/passes/structured_control_flow/unique_loop_indvars.h"

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace passes {

UniqueLoopIndvars::UniqueLoopIndvars() : Pass() {}

std::string UniqueLoopIndvars::name() { return "UniqueLoopIndvars"; }

void UniqueLoopIndvars::process_node(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::ControlFlowNode& node,
    std::unordered_set<std::string>& used_indvars,
    bool& applied
) {
    if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        auto indvar = loop->indvar();
        const std::string& name = indvar->get_name();

        if (used_indvars.find(name) != used_indvars.end()) {
            // Induction variable already used by another loop: rename it and
            // replace all uses within this loop (init, update, condition, body).
            auto& sdfg = builder.subject();
            std::string new_name = builder.find_new_name(name + "_");
            builder.add_container(new_name, sdfg.type(name));

            loop->replace(indvar, symbolic::symbol(new_name));
            used_indvars.insert(new_name);
            applied = true;
        } else {
            used_indvars.insert(name);
        }

        process_node(builder, loop->root(), used_indvars, applied);
    } else if (auto* while_stmt = dynamic_cast<structured_control_flow::While*>(&node)) {
        process_node(builder, while_stmt->root(), used_indvars, applied);
    } else if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); i++) {
            process_node(builder, sequence->at(i).first, used_indvars, applied);
        }
    } else if (auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            process_node(builder, if_else->at(i).first, used_indvars, applied);
        }
    }
    // Block, Break, Continue, Return: no nested loops to process.
}

bool UniqueLoopIndvars::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;
    std::unordered_set<std::string> used_indvars;
    process_node(builder, builder.subject().root(), used_indvars, applied);
    return applied;
}

} // namespace passes
} // namespace sdfg
