/**
 * @file memlet_delinearization_analysis.h
 * @brief Analysis for delinearizing memlet subsets
 *
 * This analysis attempts to delinearize memlet subsets by recovering
 * multi-dimensional structure from linearized expressions using the
 * symbolic delinearize function with block-level assumptions.
 *
 * The delinearization technique is based on the algorithm described in:
 * https://dl.acm.org/doi/10.1145/2751205.2751248
 */

#pragma once

#include <map>
#include <unordered_map>

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

typedef math::tensor::TensorLayout MemoryLayout;

struct MemoryAccess {
    std::string container; // Container name (memlet.data())
    data_flow::Subset subset; // Symbolic indices after delinearization
    MemoryLayout layout; // Inferred memory layout
    bool first_dim_bounded; // True if first dimension is bounded (Tensor/Array), false for unbounded pointers
};

/**
 * @class MemoryLayoutAnalysis
 * @brief Analysis that infers the assumed memory layouts of memlets
 *
 * This analysis attempts to infer the memory layout of memlets using
 * symbolic assumptions to interpret linearized subset expressions.
 *
 * The technique is based on the algorithm described in:
 * "Optimistic Delinearization of Parametrically Sized Arrays"
 * by Grosser et al.
 * https://dl.acm.org/doi/10.1145/2751205.2751248
 */
class MemoryLayoutAnalysis : public Analysis {
private:
    std::unordered_map<const data_flow::Memlet*, MemoryAccess> layouts_;

    // Per-loop merged layouts: keyed by (loop pointer, container name)
    std::map<std::pair<const structured_control_flow::StructuredLoop*, std::string>, MemoryLayout> loop_layouts_;

    void traverse(structured_control_flow::ControlFlowNode& node, analysis::AnalysisManager& analysis_manager);

    void process_block(structured_control_flow::Block& block, analysis::AnalysisManager& analysis_manager);

    void merge_loop_layouts(
        structured_control_flow::StructuredLoop& loop,
        const std::vector<const data_flow::Memlet*>& memlets_before,
        analysis::AnalysisManager& analysis_manager
    );

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    MemoryLayoutAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "MemoryLayoutAnalysis"; }

    /**
     * @brief Get the inferred memory layout for a memlet
     * @param memlet The memlet to query
     * @return A pointer to the inferred memory layout information if inference was successful, nullptr otherwise
     */
    const MemoryAccess* get(const data_flow::Memlet& memlet) const;

    /**
     * @brief Get the inferred memory layout for a container at a specific loop level
     * @param loop The loop to query
     * @param container The container name
     * @return A pointer to the memory layout at that loop level, nullptr if not available
     */
    const MemoryLayout* get(const structured_control_flow::StructuredLoop& loop, const std::string& container) const;
};

} // namespace analysis
} // namespace sdfg
