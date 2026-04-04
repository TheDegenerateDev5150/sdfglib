#include "sdfg/transformations/in_local_storage.h"

#include <cassert>
#include <cstddef>
#include <string>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/utils.h"
#include "sdfg/types/array.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace transformations {

InLocalStorage::InLocalStorage(structured_control_flow::StructuredLoop& loop, std::string container)
    : loop_(loop), container_(std::move(container)) {}

std::string InLocalStorage::name() const { return "InLocalStorage"; }

bool InLocalStorage::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& body = this->loop_.root();

    // Criterion: Container must exist in the SDFG
    if (!sdfg.exists(this->container_)) {
        return false;
    }

    // Criterion: Container must be an array type
    auto& container_type = sdfg.type(this->container_);
    if (container_type.type_id() != types::TypeID::Pointer && container_type.type_id() != types::TypeID::Array) {
        return false;
    }

    // Criterion: Check if container is used in the loop
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, body);
    if (body_users.uses(this->container_).empty()) {
        return false;
    }

    // Criterion: Container must be READ-ONLY within the loop (no writes)
    // This is what distinguishes InLocalStorage from OutLocalStorage
    if (!body_users.writes(this->container_).empty()) {
        return false;
    }

    // Criterion: All accesses must have the same dimensionality
    auto accesses = body_users.uses(this->container_);
    auto first_access = accesses.at(0);
    auto first_subsets = first_access->subsets();
    if (first_subsets.empty()) {
        return false;
    }
    auto& first_subset = first_subsets.at(0);
    if (first_subset.size() == 0) {
        return false;
    }

    for (auto* access : accesses) {
        for (auto& subset : access->subsets()) {
            if (subset.size() != first_subset.size()) {
                return false;
            }
        }
    }

    // Store representative subset for later use
    access_info_.representative_subset = first_subset;

    // Analyze index expressions per dimension
    // For simplicity, we require that the access pattern uses the loop indvar
    // and the buffer size equals the iteration count
    auto iteration_count = this->loop_.num_iterations();
    if (iteration_count.is_null()) {
        return false;
    }

    // For each dimension, determine if it depends on the loop indvar
    // We support 1D localization: the dimension that uses loop indvar gets iteration_count size
    access_info_.dimensions.clear();
    access_info_.bases.clear();

    bool found_loop_dim = false;
    for (size_t d = 0; d < first_subset.size(); d++) {
        auto& index_expr = first_subset.at(d);
        auto atoms = symbolic::atoms(index_expr);

        bool depends_on_indvar = false;
        for (auto& atom : atoms) {
            if (symbolic::eq(atom, this->loop_.indvar())) {
                depends_on_indvar = true;
                break;
            }
        }

        if (depends_on_indvar) {
            // This dimension is indexed by the loop variable
            // Buffer dimension = iteration count
            access_info_.dimensions.push_back(iteration_count);
            // Base = the constant part (index_expr - indvar contribution at init)
            auto base = symbolic::subs(index_expr, this->loop_.indvar(), this->loop_.init());
            access_info_.bases.push_back(base);
            found_loop_dim = true;
        } else {
            // This dimension is constant across loop iterations
            // We still need to capture it for multi-dimensional buffers
            access_info_.dimensions.push_back(symbolic::integer(1));
            access_info_.bases.push_back(index_expr);
        }
    }

    // We need at least one dimension that depends on the loop indvar
    if (!found_loop_dim) {
        return false;
    }

    return true;
}

void InLocalStorage::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (!parent) {
        throw InvalidSDFGException("InLocalStorage: Parent of loop must be a Sequence!");
    }

    // Get type information for the original container
    analysis::TypeAnalysis type_analysis(sdfg, &loop_, analysis_manager);
    auto type = type_analysis.get_outer_type(container_);
    types::Scalar scalar_type(type->primitive_type());

    // Create local buffer name
    local_name_ = "__daisy_in_local_storage_" + this->container_;

    // Create the local buffer array
    // For now, we support 1D buffer matching the iteration count
    auto iteration_count = this->loop_.num_iterations();
    types::Array array_type(scalar_type, iteration_count);
    builder.add_container(local_name_, array_type);

    // Create loop index variable for the copy loop
    auto indvar_name = "__daisy_in_local_storage_" + this->loop_.indvar()->get_name();
    types::Scalar indvar_type(sdfg.type(loop_.indvar()->get_name()).primitive_type());
    builder.add_container(indvar_name, indvar_type);
    auto indvar = symbolic::symbol(indvar_name);

    auto init = loop_.init();
    auto update = symbolic::subs(loop_.update(), loop_.indvar(), indvar);
    auto condition = symbolic::subs(loop_.condition(), loop_.indvar(), indvar);

    // Get access information from loop body
    analysis::UsersView body_users(users, loop_.root());
    auto accesses = body_users.uses(this->container_);
    auto first_access = accesses.at(0);
    auto first_subset = first_access->subsets().at(0);

    // Insert copy loop BEFORE the main loop: A_local[i] = A[...]
    auto& copy_loop = builder.add_for_before(*parent, loop_, indvar, condition, init, update, {}, loop_.debug_info());
    auto& copy_body = copy_loop.root();
    auto& copy_block = builder.add_block(copy_body);

    // Read from original container
    auto& copy_access_src = builder.add_access(copy_block, this->container_);
    // Write to local buffer
    auto& copy_access_dst = builder.add_access(copy_block, local_name_);
    // Tasklet: _out = _in
    auto& copy_tasklet = builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"});

    // Input memlet: read from original array with substituted index
    auto& copy_memlet_in =
        builder.add_computational_memlet(copy_block, copy_access_src, copy_tasklet, "_in", first_subset, *type);
    copy_memlet_in.replace(loop_.indvar(), indvar);

    // Output memlet: write to local buffer indexed by loop variable
    builder.add_computational_memlet(copy_block, copy_tasklet, "_out", copy_access_dst, {indvar}, array_type);

    // Now update all accesses in the main loop body to use the local buffer
    // Change subset to use just the loop indvar
    for (auto* user : body_users.uses(this->container_)) {
        auto element = user->element();
        if (auto memlet = dynamic_cast<data_flow::Memlet*>(element)) {
            auto subset = memlet->subset();
            subset.clear();
            subset.push_back(this->loop_.indvar());
            memlet->set_subset(subset);
        }
    }

    // Update access node edges to use local buffer type
    for (auto* user : body_users.uses(this->container_)) {
        auto element = user->element();
        if (auto access = dynamic_cast<data_flow::AccessNode*>(element)) {
            for (auto& iedge : access->get_parent().in_edges(*access)) {
                auto memlet = &iedge;
                auto subset = memlet->subset();
                subset.clear();
                subset.push_back(this->loop_.indvar());
                memlet->set_subset(subset);
                memlet->set_base_type(array_type);
            }
            for (auto& oedge : access->get_parent().out_edges(*access)) {
                auto memlet = &oedge;
                auto subset = memlet->subset();
                subset.clear();
                subset.push_back(this->loop_.indvar());
                memlet->set_subset(subset);
                memlet->set_base_type(array_type);
            }
        }
    }

    // Replace container name in the loop body
    loop_.replace(symbolic::symbol(this->container_), symbolic::symbol(local_name_));

    // Cleanup
    analysis_manager.invalidate_all();

    passes::SequenceFusion sf_pass;
    passes::DeadCFGElimination dce_pass;
    bool applies = false;
    do {
        applies = false;
        applies |= dce_pass.run(builder, analysis_manager);
        applies |= sf_pass.run(builder, analysis_manager);
    } while (applies);
}

void InLocalStorage::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw std::runtime_error("Unsupported loop type for serialization of loop: " + loop_.indvar()->get_name());
    }
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["transformation_type"] = this->name();
    j["container"] = container_;
}

InLocalStorage InLocalStorage::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    std::string container = desc["container"].get<std::string>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (!loop) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }

    return InLocalStorage(*loop, container);
}

} // namespace transformations
} // namespace sdfg
