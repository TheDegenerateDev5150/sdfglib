#include "sdfg/transformations/loop_peeling.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

#include <symengine/integer.h>

namespace sdfg {
namespace transformations {

LoopPeeling::LoopPeeling(structured_control_flow::StructuredLoop& loop) : loop_(loop) {};

std::string LoopPeeling::name() const { return "LoopPeeling"; };

/// True if `expr` is a strictly positive integer constant.
static bool is_positive_int(const symbolic::Expression& expr) {
    return expr != SymEngine::null && SymEngine::is_a<SymEngine::Integer>(*expr) &&
           SymEngine::rcp_static_cast<const SymEngine::Integer>(expr)->as_int() > 0;
}

/// Applicable when the loop has a constant-trip overapproximation (so the nest
/// can be fully unrolled) but a non-constant exact trip count (so there is a
/// dynamic boundary worth predicating). Relies on the StructuredLoop trip-count
/// helpers, which handle `<=`, offsets, strides and tile-style `min(...)` bounds.
static bool has_predicable_boundary(structured_control_flow::StructuredLoop& loop) {
    if (!loop.is_monotonic()) {
        return false;
    }
    auto approx = loop.num_iterations_approx();
    if (!is_positive_int(approx)) {
        return false;
    }
    auto exact = loop.num_iterations();
    if (exact == SymEngine::null || is_positive_int(exact)) {
        return false;
    }
    return true;
}

bool LoopPeeling::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return has_predicable_boundary(loop_);
};

void LoopPeeling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto indvar = loop_.indvar();
    auto init = loop_.init();
    auto update = loop_.update();

    // Shift the induction variable to `[0, trip*stride)` so the trip count is a
    // literal constant. clang only fully unrolls (and thus scalarizes register
    // tiles) 0-based constant-trip loops, not symbolic-offset ones such as
    // `for (k = k_tile0; k < k_tile0 + 8; ...)`. The original induction value is
    // `indvar + init`, substituted into both the guard and the body below.
    auto trip = loop_.num_iterations_approx();
    auto zero_init = symbolic::integer(0);
    auto shifted_indvar = symbolic::add(indvar, init);
    symbolic::Condition new_condition = symbolic::Lt(indvar, symbolic::mul(trip, loop_.stride()));

    // The original condition guards the body (rewritten for the shifted indvar):
    // the overapproximated range is a superset, so re-checking it reproduces
    // exactly the iterations the original loop executed.
    symbolic::Condition guard = symbolic::subs(loop_.condition(), indvar, shifted_indvar);

    auto parent = static_cast<structured_control_flow::Sequence*>(loop_.get_parent());

    // Replacement loop (same kind/indvar/init/update/schedule), inserted before the original.
    structured_control_flow::StructuredLoop* new_loop = nullptr;
    if (auto map = dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        new_loop = &builder.add_map_before(
            *parent, loop_, indvar, new_condition, zero_init, update, map->schedule_type(), loop_.debug_info()
        );
    } else if (auto reduce = dynamic_cast<structured_control_flow::Reduce*>(&loop_)) {
        new_loop = &builder.add_reduce_before(
            *parent,
            loop_,
            indvar,
            new_condition,
            zero_init,
            update,
            reduce->reductions(),
            reduce->schedule_type(),
            loop_.debug_info()
        );
    } else {
        new_loop =
            &builder.add_for_before(*parent, loop_, indvar, new_condition, zero_init, update, loop_.debug_info());
    }

    // Guard the body: single-case IfElse(guard) holding a deep copy of the original body.
    auto& if_else = builder.add_if_else(new_loop->root(), loop_.debug_info());
    auto& case_branch = builder.add_case(if_else, guard, loop_.debug_info());
    deepcopy::StructuredSDFGDeepCopy body_copier(builder, case_branch, loop_.root());
    body_copier.insert();

    // Rewrite induction-variable uses in the shifted body to the original value.
    if (!symbolic::eq(init, zero_init)) {
        case_branch.replace(indvar, shifted_indvar);
    }

    // Remove the original loop.
    builder.remove_child(*parent, parent->index(loop_));

    analysis_manager.invalidate_all();
};

void LoopPeeling::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
};

LoopPeeling LoopPeeling::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }
    return LoopPeeling(*loop);
};

} // namespace transformations
} // namespace sdfg
