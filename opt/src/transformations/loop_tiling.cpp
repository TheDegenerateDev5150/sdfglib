#include "sdfg/transformations/loop_tiling.h"

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace transformations {

LoopTiling::LoopTiling(structured_control_flow::StructuredLoop& loop, size_t tile_size)
    : loop_(loop), tile_size_(tile_size) {};

LoopTiling::LoopTiling(structured_control_flow::StructuredLoop& loop, size_t tile_size, size_t tile_size_2)
    : loop_(loop), tile_size_(tile_size), tile_size_2_(tile_size_2) {};

std::string LoopTiling::name() const { return "LoopTiling"; };

bool LoopTiling::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (this->tile_size_ <= 1) {
        return false;
    }
    if (this->tile_size_2_ == 1) {
        return false;
    }
    if (this->tile_size_2_ > 0 && this->tile_size_2_ >= this->tile_size_) {
        return false;
    }
    if (this->tile_size_2_ > 0 && this->tile_size_ % this->tile_size_2_ != 0) {
        return false;
    }
    // Criterion contiguous loop
    return loop_.is_contiguous();
};

void LoopTiling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&loop_));
    size_t index = parent->index(loop_);
    auto& transition = parent->at(index).second;

    auto indvar = loop_.indvar();

    // Step 1: Define new outer loop
    auto outer_indvar_str = builder.find_new_name(indvar->get_name() + "_tile");
    builder.add_container(outer_indvar_str, sdfg.type(loop_.indvar()->get_name()));

    auto outer_indvar = symbolic::symbol(outer_indvar_str);
    auto outer_condition = symbolic::subs(loop_.condition(), indvar, outer_indvar);
    auto outer_update = symbolic::add(outer_indvar, symbolic::integer(this->tile_size_));

    structured_control_flow::StructuredLoop* outer_loop = nullptr;
    if (auto map = dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        outer_loop = &builder.add_map_before(
            *parent,
            loop_,
            outer_indvar,
            outer_condition,
            loop_.init(),
            outer_update,
            map->schedule_type(),
            transition.assignments(),
            loop_.debug_info()
        );
    } else {
        outer_loop = &builder.add_for_before(
            *parent,
            loop_,
            outer_indvar,
            outer_condition,
            loop_.init(),
            outer_update,
            transition.assignments(),
            loop_.debug_info()
        );
    }

    // Step 2: Redefine inner loop
    auto inner_indvar = indvar;
    auto inner_init = outer_indvar;
    auto inner_condition_tile =
        symbolic::Lt(inner_indvar, symbolic::add(outer_indvar, symbolic::integer(this->tile_size_)));

    symbolic::Condition inner_condition = symbolic::And(inner_condition_tile, loop_.condition());

    auto inner_update = symbolic::add(inner_indvar, symbolic::integer(1));
    builder.update_loop(loop_, inner_indvar, inner_condition, inner_init, inner_update);

    // Step 3: Move loop into tiling loop
    transition.assignments().clear();
    builder.move_child(*parent, index + 1, outer_loop->root());

    analysis_manager.invalidate_all();
    applied_ = true;
    inner_loop_ = &loop_;
    outer_loop_ = outer_loop;

    // Two-level tiling: tile the inner (point) loop again with tile_size_2_
    if (tile_size_2_ > 0) {
        auto& inner = *inner_loop_;
        auto inner_indvar2 = inner.indvar();

        auto middle_indvar_str = builder.find_new_name(inner_indvar2->get_name() + "_tile");
        builder.add_container(middle_indvar_str, sdfg.type(inner_indvar2->get_name()));

        auto middle_indvar = symbolic::symbol(middle_indvar_str);
        auto middle_condition = symbolic::subs(inner.condition(), inner_indvar2, middle_indvar);
        auto middle_update = symbolic::add(middle_indvar, symbolic::integer(this->tile_size_2_));

        auto& scope_analysis2 = analysis_manager.get<analysis::ScopeAnalysis>();
        auto parent2 = static_cast<structured_control_flow::Sequence*>(scope_analysis2.parent_scope(&inner));
        size_t index2 = parent2->index(inner);
        auto& transition2 = parent2->at(index2).second;

        structured_control_flow::StructuredLoop* middle_loop = nullptr;
        if (auto map = dynamic_cast<structured_control_flow::Map*>(&inner)) {
            middle_loop = &builder.add_map_before(
                *parent2,
                inner,
                middle_indvar,
                middle_condition,
                inner.init(),
                middle_update,
                map->schedule_type(),
                transition2.assignments(),
                inner.debug_info()
            );
        } else {
            middle_loop = &builder.add_for_before(
                *parent2,
                inner,
                middle_indvar,
                middle_condition,
                inner.init(),
                middle_update,
                transition2.assignments(),
                inner.debug_info()
            );
        }

        // Redefine the innermost loop
        auto innermost_init = middle_indvar;
        auto innermost_condition_tile =
            symbolic::Lt(inner_indvar2, symbolic::add(middle_indvar, symbolic::integer(this->tile_size_2_)));
        symbolic::Condition innermost_condition = symbolic::And(innermost_condition_tile, inner.condition());
        auto innermost_update = symbolic::add(inner_indvar2, symbolic::integer(1));
        builder.update_loop(inner, inner_indvar2, innermost_condition, innermost_init, innermost_update);

        // Move inner loop into middle loop
        transition2.assignments().clear();
        builder.move_child(*parent2, index2 + 1, middle_loop->root());

        analysis_manager.invalidate_all();
        middle_loop_ = middle_loop;
        inner_loop_ = &inner;
    }
};

void LoopTiling::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw InvalidSDFGException("Unsupported loop type for serialization of loop: " + loop_.indvar()->get_name());
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = {{"tile_size", tile_size_}};
    if (tile_size_2_ > 0) {
        j["parameters"]["tile_size_2"] = tile_size_2_;
    }
};

LoopTiling LoopTiling::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    size_t tile_size = desc["parameters"]["tile_size"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);

    size_t tile_size_2 = 0;
    if (desc["parameters"].contains("tile_size_2")) {
        tile_size_2 = desc["parameters"]["tile_size_2"].get<size_t>();
    }

    if (tile_size_2 > 0) {
        return LoopTiling(*loop, tile_size, tile_size_2);
    }
    return LoopTiling(*loop, tile_size);
};

structured_control_flow::StructuredLoop* LoopTiling::inner_loop() {
    if (!applied_) {
        throw InvalidSDFGException("Accessing tiled loop before their creation.");
    }

    return inner_loop_;
}

structured_control_flow::StructuredLoop* LoopTiling::middle_loop() {
    if (!applied_) {
        throw InvalidSDFGException("Accessing tiled loop before their creation.");
    }
    return middle_loop_;
}

structured_control_flow::StructuredLoop* LoopTiling::outer_loop() {
    if (!applied_) {
        throw InvalidSDFGException("Accessing tiled loop before their creation.");
    }

    return outer_loop_;
}

} // namespace transformations
} // namespace sdfg
