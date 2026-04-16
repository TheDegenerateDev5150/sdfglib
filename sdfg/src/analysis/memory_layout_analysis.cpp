#include "sdfg/analysis/memory_layout_analysis.h"

#include <iostream>
#include <optional>
#include <unordered_set>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace analysis {

MemoryLayoutAnalysis::MemoryLayoutAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void MemoryLayoutAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    layouts_.clear();
    loop_layouts_.clear();
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
        memlets_before.reserve(layouts_.size());
        for (const auto& entry : layouts_) {
            memlets_before.push_back(entry.first);
        }

        traverse(loop->root(), analysis_manager);

        // Merge layouts for containers accessed within this loop
        merge_loop_layouts(*loop, memlets_before, analysis_manager);
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
                this->layouts_.emplace(&memlet, layout_info);
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
                this->layouts_.emplace(&memlet, layout_info);
                continue;
            }
            case types::TypeID::Pointer: {
                // For pointers, we attempt to delinearize the access pattern to infer the layout based
                // on assumptions from loop bounds
                auto* pointer_type = dynamic_cast<const types::Pointer*>(&memlet.base_type());
                if (pointer_type->pointee_type().type_id() != types::TypeID::Scalar) {
                    std::cerr << "[DEBUG] Skipping non-scalar pointer for " << container_name << std::endl;
                    continue; // Skip non-scalar pointers
                }

                if (subset.size() != 1) {
                    std::cerr << "[DEBUG] Skipping " << container_name << ": subset.size() = " << subset.size()
                              << " (requires 1)" << std::endl;
                    continue; // Require full linearization
                }
                auto& linearized_expr = subset.at(0);

                std::cerr << "[DEBUG] Delinearizing " << container_name << ": " << *linearized_expr << std::endl;
                std::cerr << "[DEBUG] Assumptions available:" << std::endl;
                for (const auto& [sym, assump] : assumptions) {
                    for (const auto& bound : assump.upper_bounds()) {
                        std::cerr << "  " << *sym << ": upper bound " << *bound << std::endl;
                    }
                    for (const auto& bound : assump.lower_bounds()) {
                        std::cerr << "  " << *sym << ": lower bound " << *bound << std::endl;
                    }
                }

                auto result = symbolic::delinearize(linearized_expr, assumptions);
                if (!result.success) {
                    std::cerr << "[DEBUG] Delinearization FAILED for " << container_name << std::endl;
                    continue; // Delinearization failed, skip
                }

                std::cerr << "[DEBUG] Delinearization SUCCESS for " << container_name << std::endl;
                std::cerr << "[DEBUG]   Indices: ";
                for (const auto& idx : result.indices) {
                    std::cerr << *idx << " ";
                }
                std::cerr << std::endl;
                std::cerr << "[DEBUG]   Dimensions: ";
                for (const auto& dim : result.dimensions) {
                    std::cerr << *dim << " ";
                }
                std::cerr << std::endl;
                if (!result.success) {
                    continue; // Delinearization failed, skip
                }

                // Delinearization returns N indices but only N-1 dimensions (from stride division)
                // The first dimension is unbounded - insert a placeholder that will be filled in by merge
                // Using a special symbol as placeholder for the first dimension
                symbolic::MultiExpression shape;
                shape.push_back(symbolic::symbol("__first_dim_placeholder__"));
                for (const auto& dim : result.dimensions) {
                    shape.push_back(dim);
                }

                // Store symbolic indices and dimensions with unbounded first dimension
                // The merge phase will attempt to bound the first dimension using loop assumptions
                MemoryLayout layout(shape);
                MemoryAccess layout_info{container_name, result.indices, layout, false};
                this->layouts_.emplace(&memlet, layout_info);
                continue;
            }
            default:
                continue; // Skip unsupported types
        }
    }
}

const MemoryAccess* MemoryLayoutAnalysis::get(const data_flow::Memlet& memlet) const {
    auto layout_it = layouts_.find(&memlet);
    if (layout_it == layouts_.end()) {
        return nullptr;
    }
    return &layout_it->second;
}

void MemoryLayoutAnalysis::merge_loop_layouts(
    structured_control_flow::StructuredLoop& loop,
    const std::vector<const data_flow::Memlet*>& memlets_before,
    analysis::AnalysisManager& analysis_manager
) {
    // Convert memlets_before to a set for O(1) lookup
    std::unordered_set<const data_flow::Memlet*> before_set(memlets_before.begin(), memlets_before.end());

    // Group newly added unbounded layouts by container
    std::unordered_map<std::string, std::vector<const data_flow::Memlet*>> container_groups;
    for (auto& [memlet_ptr, access] : layouts_) {
        if (before_set.find(memlet_ptr) != before_set.end()) {
            continue; // Skip memlets that existed before this loop
        }
        if (access.first_dim_bounded) {
            continue; // Skip already bounded layouts
        }
        container_groups[access.container].push_back(memlet_ptr);
    }

    // Get assumptions at loop body level (includes loop variable bounds)
    // The loop's indvar bounds are only available inside the loop body, not at the loop node itself
    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto& assumptions = assumptions_analysis.get(loop.root());

    // Process each container group
    for (auto& [container, memlets] : container_groups) {
        if (memlets.empty()) continue;

        // Collect all inner dimensions and first-dim indices for consistency check
        auto& first_access = layouts_.at(memlets[0]);
        auto& reference_shape = first_access.layout.shape();
        if (reference_shape.empty()) continue;

        // Check consistency of inner dimensions across all accesses
        bool consistent = true;
        std::vector<symbolic::Expression> first_dim_indices;
        first_dim_indices.reserve(memlets.size());

        for (const auto* memlet_ptr : memlets) {
            auto& access = layouts_.at(memlet_ptr);
            auto& shape = access.layout.shape();

            // Check inner dimensions match (all except first)
            if (shape.size() != reference_shape.size()) {
                consistent = false;
                break;
            }
            for (size_t i = 1; i < shape.size(); ++i) {
                if (!symbolic::eq(shape[i], reference_shape[i])) {
                    consistent = false;
                    break;
                }
            }
            if (!consistent) break;

            // Collect first-dimension index
            if (!access.subset.empty()) {
                first_dim_indices.push_back(access.subset[0]);
            }
        }

        if (!consistent || first_dim_indices.empty()) {
            continue; // Skip containers with inconsistent layouts
        }

        // Attempt to bound first dimension from collected indices
        // Find the maximum offset across all first-dim indices
        std::optional<symbolic::Expression> max_bound;

        for (const auto& first_idx : first_dim_indices) {
            // Extract symbols from the index expression
            auto syms = symbolic::atoms(first_idx);
            if (syms.size() != 1) {
                // Can't handle multi-symbol or constant-only indices at this level
                // Skip and let outer loop try
                continue;
            }
            auto sym = *syms.begin();

            // Check if we have assumptions for this symbol
            if (assumptions.find(sym) == assumptions.end()) {
                continue; // No bounds for this symbol at this loop level
            }

            // Get the affine coefficients: first_idx = mul * sym + add
            symbolic::SymbolVec gens = {sym};
            auto polynomial = symbolic::polynomial(first_idx, gens);
            if (polynomial == SymEngine::null) {
                continue; // Not a polynomial
            }
            auto coeffs = symbolic::affine_coefficients(polynomial, gens);
            if (coeffs.find(sym) == coeffs.end()) {
                continue;
            }

            auto mul = coeffs.at(sym);
            if (!symbolic::eq(mul, symbolic::one())) {
                continue; // Non-unit coefficient, can't simply bound
            }

            auto constant_sym = symbolic::symbol("__daisy_constant__");
            symbolic::Expression add;
            if (coeffs.count(constant_sym)) {
                add = coeffs.at(constant_sym);
            } else {
                add = symbolic::zero();
            }
            auto& sym_assumption = assumptions.at(sym);
            auto sym_bound = sym_assumption.tight_upper_bound();

            // Accessed range upper bound: sym_bound + add + 1 (exclusive)
            auto access_bound = symbolic::simplify(symbolic::add(symbolic::add(sym_bound, add), symbolic::one()));

            if (!max_bound.has_value()) {
                max_bound = access_bound;
            } else {
                // Take maximum of current max and this bound
                // For simplicity, if they differ symbolically, keep symbolic max
                if (!symbolic::eq(*max_bound, access_bound)) {
                    max_bound = symbolic::max(*max_bound, access_bound);
                }
            }
        }

        if (!max_bound.has_value()) {
            continue; // Could not bound first dimension at this loop level
        }
        auto max_bound_val = symbolic::expand(*max_bound);
        max_bound_val = symbolic::simplify(max_bound_val);

        // Store this loop's view of the container
        auto loop_shape = first_access.layout.shape();
        if (!loop_shape.empty()) {
            loop_shape[0] = max_bound_val;
        }
        loop_layouts_.insert(
            {{&loop, container}, MemoryLayout(loop_shape, first_access.layout.strides(), first_access.layout.offset())}
        );

        // Update all accesses in this container with the bounded first dimension
        for (const auto* memlet_ptr : memlets) {
            auto& access = layouts_.at(memlet_ptr);
            // Copy data before modifying
            auto container = access.container;
            auto subset = access.subset;
            auto new_shape = access.layout.shape();
            auto strides = access.layout.strides();
            auto offset = access.layout.offset();
            if (!new_shape.empty()) {
                new_shape[0] = max_bound_val;
            }
            // Update the layout with bounded first dimension
            layouts_.erase(memlet_ptr);
            layouts_
                .emplace(memlet_ptr, MemoryAccess{container, subset, MemoryLayout(new_shape, strides, offset), true});
        }
    }
}

const MemoryLayout* MemoryLayoutAnalysis::
    get(const structured_control_flow::StructuredLoop& loop, const std::string& container) const {
    auto key = std::make_pair(&loop, container);
    auto it = loop_layouts_.find(key);
    if (it == loop_layouts_.end()) {
        return nullptr;
    }
    return &it->second;
}

} // namespace analysis
} // namespace sdfg
