#include <gtest/gtest.h>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/passes/rpc/daisytuner_rpc_context.h"
#include "sdfg/passes/rpc/rpc_scheduler.h"
#include "sdfg/serializer/json_serializer.h"


class DaisytunerRpcTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        try {
            auto ctx = sdfg::passes::rpc::DaisytunerRpcContext::from_docc_config();
            sdfg::passes::rpc::register_rpc_loop_opt(ctx, "sequential", "server", true);
        } catch (const std::exception& e) {
            // Optionally print a warning or skip tests
        }
    }
};

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();
    ::testing::AddGlobalTestEnvironment(new DaisytunerRpcTestEnvironment);
    return RUN_ALL_TESTS();
}
