#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop collapse transformation
 *
 * This transformation collapses `count` tightly-nested loops into a single
 * flat loop, increasing the iteration space and exposing more parallelism.
 *
 * @note The outermost loop of the nest is passed as `loop`
 * @note `count` must be >= 2
 */
class LoopCollapse : public Transformation {
    structured_control_flow::Map& loop_;
    size_t count_;
    bool applied_ = false;

    structured_control_flow::StructuredLoop* collapsed_loop_ = nullptr;

public:
    /**
     * @brief Construct a loop collapse transformation
     * @param loop The outermost loop of the nest to collapse
     * @param count The number of loops to collapse (must be >= 2)
     */
    LoopCollapse(structured_control_flow::Map& loop, size_t count);

    /**
     * @brief Get the name of this transformation
     * @return "LoopCollapse"
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
     * @brief Apply the loop collapse transformation
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
     * @brief Deserialize a loop collapse transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static LoopCollapse from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /**
     * @brief Get the resulting collapsed loop after the transformation has been applied
     */
    structured_control_flow::StructuredLoop* collapsed_loop();
};

} // namespace transformations
} // namespace sdfg
