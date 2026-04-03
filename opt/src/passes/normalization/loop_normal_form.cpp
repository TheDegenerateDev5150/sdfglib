#include "sdfg/passes/normalization/loop_normal_form.h"

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

    transformations::LoopShift loop_shift(loop);
    if (loop_shift.can_be_applied(builder_, analysis_manager_)) {
        loop_shift.apply(builder_, analysis_manager_);
        applied = true;
    }

    transformations::LoopUnitStride loop_unit_stride(loop);
    if (loop_unit_stride.can_be_applied(builder_, analysis_manager_)) {
        loop_unit_stride.apply(builder_, analysis_manager_);
        applied = true;
    }

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
