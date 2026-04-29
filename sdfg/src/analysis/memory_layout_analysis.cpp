#include "sdfg/analysis/memory_layout_analysis.h"

#include <optional>
#include <set>
#include <unordered_set>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace analysis {

namespace {
// Collect StructuredLoop nodes that are direct children of the given node,
// stopping at loop boundaries (does not recurse into nested loops).
void collect_direct_child_loops(
    structured_control_flow::ControlFlowNode& node, std::set<const structured_control_flow::StructuredLoop*>& result
) {
    if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        result.insert(loop);
        return;
    }
    if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); i++) {
            collect_direct_child_loops(seq->at(i).first, result);
        }
    } else if (auto* ife = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < ife->size(); i++) {
            collect_direct_child_loops(ife->at(i).first, result);
        }
    } else if (auto* w = dynamic_cast<structured_control_flow::While*>(&node)) {
        collect_direct_child_loops(w->root(), result);
    }
}
} // namespace

MemoryLayoutAnalysis::MemoryLayoutAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void MemoryLayoutAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    accesses_.clear();
    tiles_.clear();
    traverse(sdfg_.root(), analysis_manager);
}

void MemoryLayoutAnalysis::
    traverse(structured_control_flow::ControlFlowNode& node, analysis::AnalysisManager& analysis_manager) {
    if (auto block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        process_block(*block, analysis_manager);
    } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); i++) {
            traverse(sequence->at(i).first, analysis_manager);
        }
    } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            traverse(if_else->at(i).first, analysis_manager);
        }
    } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(&node)) {
        traverse(while_stmt->root(), analysis_manager);
    } else if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        // Snapshot current memlets before traversing loop body
        std::vector<const data_flow::Memlet*> memlets_before;
        memlets_before.reserve(accesses_.size());
        for (const auto& entry : accesses_) {
            memlets_before.push_back(entry.first);
        }

        // Snapshot tile keys before traversal
        std::set<std::pair<const structured_control_flow::StructuredLoop*, std::string>> tiles_before;
        for (const auto& entry : tiles_) {
            tiles_before.insert(entry.first);
        }

        traverse(loop->root(), analysis_manager);

        // Merge layouts for containers accessed within this loop
        merge_loop_layouts(*loop, memlets_before, tiles_before, analysis_manager);
    }
    // Break, Continue, Return nodes don't contain blocks
}

void MemoryLayoutAnalysis::
    process_block(structured_control_flow::Block& block, analysis::AnalysisManager& analysis_manager) {
    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto& assumptions = assumptions_analysis.get(block);

    auto& dfg = block.dataflow();
    for (auto& memlet : dfg.edges()) {
        const auto& subset = memlet.subset();
        if (subset.empty()) {
            continue;
        }

        // Get container name from the AccessNode (either src or dst)
        std::string container_name;
        if (auto* access = dynamic_cast<const data_flow::AccessNode*>(&memlet.src())) {
            container_name = access->data();
        } else if (auto* access = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst())) {
            container_name = access->data();
        } else {
            continue; // Skip memlets without AccessNode
        }

        auto& base_type = memlet.base_type();
        switch (base_type.type_id()) {
            case types::TypeID::Scalar:
            case types::TypeID::Structure:
                continue; // Skip scalars and structures
            case types::TypeID::Tensor: {
                // Tensor types already contain layout information, so we can directly store it without delinearization
                auto& tensor_type = dynamic_cast<const types::Tensor&>(memlet.base_type());

                MemoryLayout layout(tensor_type.shape(), tensor_type.strides(), tensor_type.offset());
                MemoryAccess layout_info{container_name, subset, layout, true};
                this->accesses_.emplace(&memlet, layout_info);
                continue;
            }
            case types::TypeID::Array: {
                // Arrays are c-like stack array, so we can infer a simple row-major layout without needing
                // delinearization
                auto* array_type = dynamic_cast<const types::Array*>(&memlet.base_type());
                symbolic::MultiExpression shape = {array_type->num_elements()};
                while (array_type->element_type().type_id() == types::TypeID::Array) {
                    array_type = dynamic_cast<const types::Array*>(&array_type->element_type());
                }
                if (array_type->element_type().type_id() != types::TypeID::Scalar) {
                    continue; // Skip non-scalar arrays
                }

                MemoryLayout layout(shape);
                MemoryAccess layout_info{container_name, subset, layout, true};
                this->accesses_.emplace(&memlet, layout_info);
                continue;
            }
            case types::TypeID::Pointer: {
                // For pointers, we attempt to delinearize the access pattern to infer the layout based
                // on assumptions from loop bounds
                auto* pointer_type = dynamic_cast<const types::Pointer*>(&memlet.base_type());
                if (pointer_type->pointee_type().type_id() != types::TypeID::Scalar) {
                    continue; // Skip non-scalar pointers
                }

                if (subset.size() != 1) {
                    continue; // Require full linearization
                }
                auto& linearized_expr = subset.at(0);

                auto result = symbolic::delinearize(linearized_expr, assumptions);
                if (!result.success) {
                    continue; // Delinearization failed, skip
                }

                // Delinearization returns N indices but only N-1 dimensions (from stride division)
                // The first dimension is unbounded - insert a placeholder that will be filled in by merge
                // Using a special symbol as placeholder for the first dimension
                symbolic::MultiExpression shape;
                shape.push_back(symbolic::symbol("__unbounded__"));
                for (const auto& dim : result.dimensions) {
                    shape.push_back(dim);
                }

                // Store symbolic indices and dimensions with unbounded first dimension
                // The merge phase will attempt to bound the first dimension using loop assumptions
                MemoryLayout layout(shape);
                MemoryAccess layout_info{container_name, result.indices, layout, false};
                this->accesses_.emplace(&memlet, layout_info);
                continue;
            }
            default:
                continue; // Skip unsupported types
        }
    }
}

const MemoryAccess* MemoryLayoutAnalysis::access(const data_flow::Memlet& memlet) const {
    auto layout_it = accesses_.find(&memlet);
    if (layout_it == accesses_.end()) {
        return nullptr;
    }
    return &layout_it->second;
}

void MemoryLayoutAnalysis::merge_loop_layouts(
    structured_control_flow::StructuredLoop& loop,
    const std::vector<const data_flow::Memlet*>& memlets_before,
    const std::set<std::pair<const structured_control_flow::StructuredLoop*, std::string>>& tiles_before,
    analysis::AnalysisManager& analysis_manager
) {
    // Convert memlets_before to a set for O(1) lookup
    std::unordered_set<const data_flow::Memlet*> before_set(memlets_before.begin(), memlets_before.end());

    // Group all new accesses by container
    std::unordered_map<std::string, std::vector<const data_flow::Memlet*>> all_container_groups;
    for (auto& [memlet_ptr, acc] : accesses_) {
        if (before_set.find(memlet_ptr) != before_set.end()) {
            continue;
        }
        all_container_groups[acc.container].push_back(memlet_ptr);
    }

    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto& assumptions = assumptions_analysis.get(loop.root());
    // Start with SDFG-level parameters (read-only arguments like N, M)
    // then add any additional constant symbols from loop assumptions
    symbolic::SymbolSet parameters = assumptions_analysis.parameters();
    for (auto& entry : assumptions) {
        if (symbolic::eq(entry.first, loop.indvar())) {
            continue; // Skip induction variable itself
        }

        if (entry.second.constant()) {
            parameters.insert(entry.first);
        }
    }

    // Find direct child loops of this loop (not grandchildren)
    std::set<const structured_control_flow::StructuredLoop*> direct_child_loops;
    collect_direct_child_loops(loop.root(), direct_child_loops);

    for (auto& [container, memlets] : all_container_groups) {
        if (memlets.empty()) continue;

        // Find inner tiles from direct child loops only
        std::vector<const MemoryTile*> inner_tiles;
        for (auto& [key, tile] : tiles_) {
            if (tiles_before.count(key) > 0) continue;
            if (key.second != container) continue;
            if (direct_child_loops.count(key.first) == 0) continue;
            inner_tiles.push_back(&tile);
        }

        size_t ndims = 0;
        MemoryLayout reference_layout({symbolic::one()});
        // Separate min/max index lists to avoid unnecessary symbolic min/max
        std::vector<std::vector<symbolic::Expression>> min_indices;
        std::vector<std::vector<symbolic::Expression>> max_indices;

        if (!inner_tiles.empty()) {
            // Use inner tile min/max as representative values
            // Inner tiles have already resolved inner loop variables to their bounds
            ndims = inner_tiles[0]->min_subset.size();
            reference_layout = inner_tiles[0]->layout;
            min_indices.resize(ndims);
            max_indices.resize(ndims);

            for (const auto* tile : inner_tiles) {
                if (tile->min_subset.size() != ndims) continue;
                for (size_t d = 0; d < ndims; ++d) {
                    min_indices[d].push_back(tile->min_subset[d]);
                    max_indices[d].push_back(tile->max_subset[d]);
                }
            }
        } else {
            // Use raw access indices (no inner tiles available)
            auto& first_access = accesses_.at(memlets[0]);
            auto& reference_shape = first_access.layout.shape();
            ndims = reference_shape.size();
            reference_layout = first_access.layout;
            min_indices.resize(ndims);
            max_indices.resize(ndims);

            bool consistent = true;
            for (const auto* memlet_ptr : memlets) {
                auto& acc = accesses_.at(memlet_ptr);
                auto& shape = acc.layout.shape();

                if (shape.size() != ndims) {
                    consistent = false;
                    break;
                }
                // Check inner dimensions match (all except first which may be unbounded)
                for (size_t d = 1; d < ndims; ++d) {
                    if (!symbolic::eq(shape[d], reference_shape[d])) {
                        consistent = false;
                        break;
                    }
                }
                if (!consistent) break;

                // Collect indices for each dimension
                if (acc.subset.size() != ndims) {
                    consistent = false;
                    break;
                }
                for (size_t d = 0; d < ndims; ++d) {
                    min_indices[d].push_back(acc.subset[d]);
                    max_indices[d].push_back(acc.subset[d]);
                }
            }

            if (!consistent) continue;
        }

        if (ndims == 0) continue;

        // Compute min/max bounds for each dimension
        data_flow::Subset min_subset;
        data_flow::Subset max_subset;
        bool all_bounded = true;

        for (size_t d = 0; d < ndims; ++d) {
            symbolic::Expression dim_min = SymEngine::null;
            symbolic::Expression dim_max = SymEngine::null;

            // Compute dim_min from min_indices
            for (const auto& idx : min_indices[d]) {
                auto lb = symbolic::minimum(idx, parameters, assumptions, true);
                if (lb.is_null()) {
                    lb = symbolic::minimum(idx, parameters, assumptions, false);
                }
                if (lb.is_null()) {
                    all_bounded = false;
                    break;
                }
                if (dim_min.is_null()) {
                    dim_min = lb;
                } else {
                    dim_min = symbolic::min(dim_min, lb);
                }
            }
            if (!all_bounded) break;

            // Compute dim_max from max_indices
            for (const auto& idx : max_indices[d]) {
                auto ub = symbolic::maximum(idx, parameters, assumptions, true);
                if (ub.is_null()) {
                    ub = symbolic::maximum(idx, parameters, assumptions, false);
                }
                if (ub.is_null()) {
                    all_bounded = false;
                    break;
                }
                if (dim_max.is_null()) {
                    dim_max = ub;
                } else {
                    dim_max = symbolic::max(dim_max, ub);
                }
            }
            if (!all_bounded) break;

            min_subset.push_back(symbolic::simplify(dim_min));
            max_subset.push_back(symbolic::simplify(dim_max));
        }

        if (!all_bounded) continue;

        // Store this loop's tile with the original memory layout
        tiles_.insert({{&loop, container}, MemoryTile{container, min_subset, max_subset, reference_layout, true}});
    }
}

const MemoryTile* MemoryLayoutAnalysis::
    tile(const structured_control_flow::StructuredLoop& loop, const std::string& container) const {
    auto key = std::make_pair(&loop, container);
    auto it = tiles_.find(key);
    if (it == tiles_.end()) {
        return nullptr;
    }
    return &it->second;
}

symbolic::MultiExpression MemoryTile::extents() const {
    symbolic::MultiExpression result;
    for (size_t d = 0; d < min_subset.size(); ++d) {
        result.push_back(symbolic::simplify(symbolic::add(symbolic::sub(max_subset[d], min_subset[d]), symbolic::one()))
        );
    }
    return result;
}

std::pair<symbolic::Expression, symbolic::Expression> MemoryTile::contiguous_range() const {
    auto& strides = layout.strides();
    auto first = layout.offset();
    auto last = layout.offset();
    for (size_t d = 0; d < min_subset.size(); ++d) {
        first = symbolic::add(first, symbolic::mul(strides[d], min_subset[d]));
        last = symbolic::add(last, symbolic::mul(strides[d], max_subset[d]));
    }
    return {symbolic::simplify(first), symbolic::simplify(last)};
}

} // namespace analysis
} // namespace sdfg
