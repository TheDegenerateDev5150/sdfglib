#include "sdfg/passes/normalization/loop_normal_form.h"

#include "sdfg/transformations/loop_condition_normalize.h"
#include "sdfg/transformations/loop_rotate.h"
#include "sdfg/transformations/loop_shift.h"
#include "sdfg/transformations/loop_unit_stride.h"

namespace sdfg {
namespace passes {
namespace normalization {

LoopNormalForm::LoopNormalForm(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {};

bool LoopNormalForm::accept(structured_control_flow::For& loop) { return this->apply(loop); };

bool LoopNormalForm::accept(structured_control_flow::Map& loop) { return this->apply(loop); };

bool LoopNormalForm::apply(structured_control_flow::StructuredLoop& loop) {
    bool applied = false;

    // Step 1: Shift loop to start from 0
    transformations::LoopShift loop_shift(loop);
    if (loop_shift.can_be_applied(builder_, analysis_manager_)) {
        loop_shift.apply(builder_, analysis_manager_);
        applied = true;
    }

    // Step 2: Convert non-unit stride to unit stride
    transformations::LoopUnitStride loop_unit_stride(loop);
    if (loop_unit_stride.can_be_applied(builder_, analysis_manager_)) {
        loop_unit_stride.apply(builder_, analysis_manager_);
        applied = true;
    }

    // Step 3: Convert != conditions to < or > (requires unit stride)
    transformations::LoopConditionNormalize loop_cond_normalize(loop);
    if (loop_cond_normalize.can_be_applied(builder_, analysis_manager_)) {
        loop_cond_normalize.apply(builder_, analysis_manager_);
        applied = true;
    }

    // Step 4: Convert negative stride to positive (requires < or > conditions)
    transformations::LoopRotate loop_rotate(loop);
    if (loop_rotate.can_be_applied(builder_, analysis_manager_)) {
        loop_rotate.apply(builder_, analysis_manager_);
        applied = true;
    }

    return applied;
};


} // namespace normalization
} // namespace passes
} // namespace sdfg
