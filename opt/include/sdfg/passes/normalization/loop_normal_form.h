#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {
namespace normalization {

class LoopNormalForm : public visitor::NonStoppingStructuredSDFGVisitor {
    bool apply(structured_control_flow::StructuredLoop& node);

public:
    LoopNormalForm(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "LoopNormalForm"; };

    bool accept(structured_control_flow::For& node) override;

    bool accept(structured_control_flow::Map& node) override;
};

typedef VisitorPass<LoopNormalForm> LoopNormalFormPass;

} // namespace normalization
} // namespace passes
} // namespace sdfg
