#pragma once

#include <sdfg/passes/pass.h>
#include <string>
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/rpc/rpc_context.h"

namespace sdfg {
namespace transformations {
class Recorder;
}
namespace passes {
namespace scheduler {

class RPCSchedulingPass : public Pass {
private:
    std::shared_ptr<rpc::RpcContext> rpc_context_;
    std::string target_, category_;
    sdfg::PassReportConsumer* report_ = nullptr;

public:
    RPCSchedulingPass(
        std::shared_ptr<rpc::RpcContext> rpc_context,
        std::string target,
        std::string category,
        sdfg::PassReportConsumer* report
    )
        : rpc_context_(rpc_context), target_(target), category_(category), report_(report) {}
    ~RPCSchedulingPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override { return "RPCSchedulingPass"; }
};


} // namespace scheduler
} // namespace passes
} // namespace sdfg
