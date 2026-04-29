#include "sdfg/transformations/out_local_storage.h"

#include <cstddef>
#include <string>

#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace transformations {

OutLocalStorage::OutLocalStorage(structured_control_flow::StructuredLoop& loop, const data_flow::AccessNode& access_node)
    : loop_(loop), access_node_(access_node), container_(access_node.data()) {};

std::string OutLocalStorage::name() const { return "OutLocalStorage"; };

bool OutLocalStorage::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& body = this->loop_.root();

    tile_info_ = TileInfo{};

    // Criterion: Container must exist
    if (!sdfg.exists(this->container_)) {
        return false;
    }

    auto& type = sdfg.type(this->container_);

    // Criterion: Container must be used in the loop body
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, body);
    if (body_users.uses(this->container_).empty()) {
        return false;
    }

    // Criterion: Container must have writes (this is OutLocalStorage, not InLocalStorage)
    if (body_users.writes(this->container_).empty()) {
        return false;
    }

    // Determine if container is also read (read-write vs write-only)
    tile_info_.has_read = !body_users.reads(this->container_).empty();

    // Handle scalar containers: no tile needed, dimensions stay empty
    if (type.type_id() == types::TypeID::Scalar) {
        return true;
    }

    // For Array/Pointer types: use MemoryLayoutAnalysis tile API
    if (type.type_id() != types::TypeID::Pointer && type.type_id() != types::TypeID::Array) {
        return false;
    }

    auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();
    auto* tile = mla.tile(loop_, this->container_);
    if (!tile) {
        return false;
    }

    // Get overapproximated extents (integer upper bounds)
    auto extents = tile->extents_approx();
    if (extents.empty()) {
        return false;
    }

    // Criterion: All extents must be provably integer
    for (size_t idx = 0; idx < extents.size(); idx++) {
        auto& ext = extents[idx];
        if (!SymEngine::is_a<SymEngine::Integer>(*ext)) {
            return false;
        }
    }

    // Store tile info
    tile_info_.dimensions = extents;
    tile_info_.bases = tile->min_subset;
    tile_info_.strides =
        std::vector<symbolic::Expression>(tile->layout.strides().begin(), tile->layout.strides().end());
    tile_info_.offset = tile->layout.offset();

    return true;
}

void OutLocalStorage::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (!parent) {
        throw InvalidSDFGException("OutLocalStorage: Parent of loop must be a Sequence!");
    }

    // Get type information
    auto& type = sdfg.type(this->container_);
    types::Scalar scalar_type(type.primitive_type());

    // Create local buffer name
    local_name_ = "__daisy_out_local_storage_" + this->container_;

    // ========================================================================
    // SCALAR PATH: tile_info_.dimensions is empty
    // ========================================================================
    if (tile_info_.dimensions.empty()) {
        // Create scalar local buffer
        builder.add_container(local_name_, scalar_type);

        // Get the access subset from the first user (all scalar, so empty subset)
        analysis::UsersView body_users(users, loop_.root());
        auto accesses = body_users.uses(this->container_);
        auto first_access = accesses.at(0);
        auto first_subset = first_access->subsets().at(0);

        // Init block (copy from container to local) - before loop
        if (tile_info_.has_read) {
            auto& init_block = builder.add_block_before(*parent, loop_, {}, loop_.debug_info());
            auto& init_src = builder.add_access(init_block, this->container_);
            auto& init_dst = builder.add_access(init_block, local_name_);
            auto& init_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
            builder.add_computational_memlet(init_block, init_src, init_tasklet, "_in", first_subset, type);
            builder.add_computational_memlet(init_block, init_tasklet, "_out", init_dst, {}, scalar_type);
        }

        // Writeback block (copy from local to container) - after loop
        {
            auto& wb_block = builder.add_block_after(*parent, loop_, {}, loop_.debug_info());
            auto& wb_src = builder.add_access(wb_block, local_name_);
            auto& wb_dst = builder.add_access(wb_block, this->container_);
            auto& wb_tasklet = builder.add_tasklet(wb_block, data_flow::TaskletCode::assign, "_out", {"_in"});
            builder.add_computational_memlet(wb_block, wb_src, wb_tasklet, "_in", {}, scalar_type);
            builder.add_computational_memlet(wb_block, wb_tasklet, "_out", wb_dst, first_subset, type);
        }

        // Rewrite body accesses to use scalar local
        for (auto* user : body_users.uses(this->container_)) {
            auto element = user->element();
            if (auto access = dynamic_cast<data_flow::AccessNode*>(element)) {
                for (auto& iedge : access->get_parent().in_edges(*access)) {
                    auto memlet = &iedge;
                    memlet->set_subset({});
                    memlet->set_base_type(scalar_type);
                }
                for (auto& oedge : access->get_parent().out_edges(*access)) {
                    auto memlet = &oedge;
                    memlet->set_subset({});
                    memlet->set_base_type(scalar_type);
                }
            }
        }

        // Replace container name in the loop body
        loop_.replace(symbolic::symbol(this->container_), symbolic::symbol(local_name_));
    }
    // ========================================================================
    // ARRAY PATH: tile_info_.dimensions is non-empty
    // ========================================================================
    else {
        // Collect varying dimensions (extent > 1) and compute buffer layout
        std::vector<size_t> varying_dims;
        std::vector<symbolic::Expression> dim_sizes;
        for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
            auto& dim_size = tile_info_.dimensions.at(d);
            if (!symbolic::eq(dim_size, symbolic::integer(1))) {
                varying_dims.push_back(d);
                dim_sizes.push_back(dim_size);
            }
        }

        // Compute total buffer size
        symbolic::Expression total_size = symbolic::integer(1);
        for (auto& ds : dim_sizes) {
            total_size = symbolic::mul(total_size, ds);
        }

        // Create the local buffer
        types::Array buffer_type(scalar_type, total_size);
        builder.add_container(local_name_, buffer_type);

        // Helper: build linearized local index from per-dimension indices
        auto linearize = [&](const std::vector<symbolic::Symbol>& indvars) -> symbolic::Expression {
            symbolic::Expression linear_idx = symbolic::integer(0);
            symbolic::Expression stride = symbolic::integer(1);
            for (int i = indvars.size() - 1; i >= 0; i--) {
                linear_idx = symbolic::add(linear_idx, symbolic::mul(indvars[i], stride));
                stride = symbolic::mul(stride, dim_sizes[i]);
            }
            return linear_idx;
        };

        // Helper: build source subset (base[d] + copy_indvar[d]) for original container
        // For Pointer types: re-linearize to a single expression using layout strides
        // For Array types: produce multi-dimensional subset
        bool is_pointer = (type.type_id() == types::TypeID::Pointer);
        auto build_original_subset = [&](const std::vector<symbolic::Symbol>& copy_indvars) -> data_flow::Subset {
            // Build per-dimension indices: base[d] + indvar[d] for varying, base[d] for constant
            std::vector<symbolic::Expression> full_indices;
            size_t var_idx = 0;
            for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
                if (!symbolic::eq(tile_info_.dimensions.at(d), symbolic::integer(1))) {
                    full_indices.push_back(symbolic::add(tile_info_.bases.at(d), copy_indvars.at(var_idx++)));
                } else {
                    full_indices.push_back(tile_info_.bases.at(d));
                }
            }

            if (is_pointer) {
                // Linearize: offset + sum(stride[d] * index[d])
                symbolic::Expression linear = tile_info_.offset;
                for (size_t d = 0; d < full_indices.size(); d++) {
                    linear = symbolic::add(linear, symbolic::mul(tile_info_.strides.at(d), full_indices.at(d)));
                }
                return {linear};
            } else {
                // Multi-dimensional subset for Array types
                return data_flow::Subset(full_indices.begin(), full_indices.end());
            }
        };

        // ==================================================================
        // Create INIT loops (copy from container to tile) - before target loop
        // Only if the container is also read (read-write pattern)
        // ==================================================================
        if (tile_info_.has_read) {
            std::vector<symbolic::Symbol> init_indvars;
            structured_control_flow::Sequence* init_scope = parent;
            bool first_init_loop = true;

            for (size_t i = 0; i < varying_dims.size(); i++) {
                size_t d = varying_dims[i];
                auto indvar_name = "__daisy_ols_init_" + this->container_ + "_d" + std::to_string(d);
                types::Scalar indvar_type(types::PrimitiveType::UInt64);
                builder.add_container(indvar_name, indvar_type);
                auto indvar = symbolic::symbol(indvar_name);
                init_indvars.push_back(indvar);

                auto init = symbolic::integer(0);
                auto condition = symbolic::Lt(indvar, dim_sizes[i]);
                auto update = symbolic::add(indvar, symbolic::integer(1));

                if (first_init_loop) {
                    auto& init_loop =
                        builder
                            .add_for_before(*init_scope, loop_, indvar, condition, init, update, {}, loop_.debug_info());
                    init_scope = &init_loop.root();
                    first_init_loop = false;
                } else {
                    auto& init_loop =
                        builder.add_for(*init_scope, indvar, condition, init, update, {}, loop_.debug_info());
                    init_scope = &init_loop.root();
                }
            }

            // Create init copy block
            auto& init_block = builder.add_block(*init_scope);
            auto& init_src = builder.add_access(init_block, this->container_);
            auto& init_dst = builder.add_access(init_block, local_name_);
            auto& init_tasklet = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});

            auto init_src_subset = build_original_subset(init_indvars);
            data_flow::Subset init_dst_subset = {linearize(init_indvars)};

            builder.add_computational_memlet(init_block, init_src, init_tasklet, "_in", init_src_subset, type);
            builder.add_computational_memlet(init_block, init_tasklet, "_out", init_dst, init_dst_subset, buffer_type);
        }

        // ==================================================================
        // Create WRITEBACK loops (copy from tile to container) - after loop
        // ==================================================================
        {
            std::vector<symbolic::Symbol> wb_indvars;
            structured_control_flow::Sequence* wb_scope = parent;
            bool first_wb_loop = true;

            for (size_t i = 0; i < varying_dims.size(); i++) {
                size_t d = varying_dims[i];
                auto indvar_name = "__daisy_ols_wb_" + this->container_ + "_d" + std::to_string(d);
                types::Scalar indvar_type(types::PrimitiveType::UInt64);
                builder.add_container(indvar_name, indvar_type);
                auto indvar = symbolic::symbol(indvar_name);
                wb_indvars.push_back(indvar);

                auto init = symbolic::integer(0);
                auto condition = symbolic::Lt(indvar, dim_sizes[i]);
                auto update = symbolic::add(indvar, symbolic::integer(1));

                if (first_wb_loop) {
                    auto& wb_loop =
                        builder.add_for_after(*wb_scope, loop_, indvar, condition, init, update, {}, loop_.debug_info());
                    wb_scope = &wb_loop.root();
                    first_wb_loop = false;
                } else {
                    auto& wb_loop = builder.add_for(*wb_scope, indvar, condition, init, update, {}, loop_.debug_info());
                    wb_scope = &wb_loop.root();
                }
            }

            // Create writeback copy block
            auto& wb_block = builder.add_block(*wb_scope);
            auto& wb_src = builder.add_access(wb_block, local_name_);
            auto& wb_dst = builder.add_access(wb_block, this->container_);
            auto& wb_tasklet = builder.add_tasklet(wb_block, data_flow::TaskletCode::assign, "_out", {"_in"});

            data_flow::Subset wb_src_subset = {linearize(wb_indvars)};
            auto wb_dst_subset = build_original_subset(wb_indvars);

            builder.add_computational_memlet(wb_block, wb_src, wb_tasklet, "_in", wb_src_subset, buffer_type);
            builder.add_computational_memlet(wb_block, wb_tasklet, "_out", wb_dst, wb_dst_subset, type);
        }

        // ==================================================================
        // Update accesses in the main loop to use the local buffer
        // ==================================================================
        analysis::UsersView body_users(users, loop_.root());
        auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();

        for (auto* user : body_users.uses(this->container_)) {
            auto element = user->element();
            if (auto memlet = dynamic_cast<data_flow::Memlet*>(element)) {
                // Use MemoryLayoutAnalysis to get the delinearized access
                auto* access = mla.access(*memlet);
                if (access && access->subset.size() == tile_info_.dimensions.size()) {
                    // Compute local index: linearize (access[d] - base[d]) for varying dims
                    std::vector<symbolic::Expression> local_indices;
                    for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
                        if (!symbolic::eq(tile_info_.dimensions.at(d), symbolic::integer(1))) {
                            local_indices.push_back(symbolic::sub(access->subset.at(d), tile_info_.bases.at(d)));
                        }
                    }

                    // Linearize
                    symbolic::Expression linear_idx = symbolic::integer(0);
                    symbolic::Expression stride = symbolic::integer(1);
                    for (int i = local_indices.size() - 1; i >= 0; i--) {
                        linear_idx = symbolic::add(linear_idx, symbolic::mul(local_indices[i], stride));
                        stride = symbolic::mul(stride, dim_sizes[i]);
                    }

                    memlet->set_subset({linear_idx});
                    memlet->set_base_type(buffer_type);
                } else {
                    // Fallback: subtract bases from raw subset
                    auto& old_subset = memlet->subset();
                    if (old_subset.size() == tile_info_.dimensions.size()) {
                        std::vector<symbolic::Expression> local_indices;
                        for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
                            if (!symbolic::eq(tile_info_.dimensions.at(d), symbolic::integer(1))) {
                                local_indices.push_back(symbolic::sub(old_subset.at(d), tile_info_.bases.at(d)));
                            }
                        }

                        symbolic::Expression linear_idx = symbolic::integer(0);
                        symbolic::Expression stride = symbolic::integer(1);
                        for (int i = local_indices.size() - 1; i >= 0; i--) {
                            linear_idx = symbolic::add(linear_idx, symbolic::mul(local_indices[i], stride));
                            stride = symbolic::mul(stride, dim_sizes[i]);
                        }

                        memlet->set_subset({linear_idx});
                        memlet->set_base_type(buffer_type);
                    }
                }
            }
        }

        // Replace container name in the loop body
        loop_.replace(symbolic::symbol(this->container_), symbolic::symbol(local_name_));
    }

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
};

void OutLocalStorage::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw std::runtime_error("Unsupported loop type for serialization of loop: " + loop_.indvar()->get_name());
    }
    j["subgraph"] = {
        {"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}},
        {"1", {{"element_id", this->access_node_.element_id()}, {"type", "access_node"}}}
    };
    j["transformation_type"] = this->name();
};

OutLocalStorage OutLocalStorage::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);

    auto access_node = dynamic_cast<
        data_flow::AccessNode*>(builder.find_element_by_id(desc.at("subgraph").at("1").at("element_id").get<size_t>()));
    if (!access_node) {
        throw InvalidTransformationDescriptionException(
            "Access node with ID " + std::to_string(desc.at("subgraph").at("1").at("element_id").get<size_t>()) +
            " not found."
        );
    }

    return OutLocalStorage(*loop, *access_node);
};

} // namespace transformations
} // namespace sdfg
