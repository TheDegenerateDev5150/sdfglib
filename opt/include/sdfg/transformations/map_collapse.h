#pragma once

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Map collapse transformation
 *
 * This transformation collapses tightly-nested maps into a single flat map,
 * increasing the iteration space and exposing more parallelism.
 *
 * Two modes are supported:
 *  - Perfect nesting: `count` perfectly-nested maps are flattened into one map
 *    using a product iteration space and modulo/division index recovery.
 *  - Imperfect nesting (CUDA-kernel style, `count == 2`): the outer map's body
 *    may contain several sibling maps and other "skipped" control-flow elements.
 *    The flattened inner extent is the maximum of all sibling-map bounds; each
 *    sibling map body is guarded by `inner_idx < bound_i` and each skipped
 *    element is guarded by `inner_idx == 0` (executed once per outer iteration).
 *
 * @note The outermost map of the nest is passed as `loop`
 * @note `count` must be >= 2
 */
class MapCollapse : public Transformation {
    structured_control_flow::Map& loop_;
    size_t count_;
    bool applied_ = false;

    structured_control_flow::Map* collapsed_loop_ = nullptr;

    /// @brief Check whether `count_` perfectly-nested maps can be collapsed.
    bool check_perfect_nest();

    /// @brief Check whether the outer map can be collapsed with the (possibly
    /// imperfectly nested) maps directly contained in its body.
    bool check_imperfect();

    /// @brief Whether a direct-child map can participate in the flattened
    /// iteration space (contiguous, closed-form bound independent of the outer
    /// induction variable).
    bool is_collapsible_inner_map(structured_control_flow::Map& map, const symbolic::Symbol& outer_indvar);

    /// @brief Apply the perfectly-nested collapse (product iteration space).
    void apply_perfect(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    /// @brief Apply the imperfect (CUDA-style) collapse using a max-bound inner
    /// extent and guards for skipped elements.
    void apply_imperfect(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

public:
    /**
     * @brief Construct a map collapse transformation
     * @param loop The outermost map of the nest to collapse
     * @param count The number of maps to collapse (must be >= 2)
     */
    MapCollapse(structured_control_flow::Map& loop, size_t count);

    /**
     * @brief Get the name of this transformation
     * @return "MapCollapse"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the transformation can be applied safely
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    /**
     * @brief Apply the map collapse transformation
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     */
    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Serialize this transformation to JSON
     * @param j JSON object to populate
     */
    virtual void to_json(nlohmann::json& j) const override;

    /**
     * @brief Deserialize a map collapse transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static MapCollapse from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /**
     * @brief Get the resulting collapsed map after the transformation has been applied
     */
    structured_control_flow::Map* collapsed_loop();
};

} // namespace transformations
} // namespace sdfg
