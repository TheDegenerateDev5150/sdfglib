#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/async_copy_node.h"
#include "sdfg/serializer/json_serializer.h"

using namespace sdfg;
using data_flow::CpAsyncCopyNode;
using data_flow::PipelineCommitNode;
using data_flow::PipelineWaitNode;

namespace {
builder::StructuredSDFGBuilder make_builder() { return builder::StructuredSDFGBuilder("async_test", FunctionType_CPU); }
} // namespace

TEST(AsyncCopyNodeTest, ConstructAndProperties) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());

    auto& copy = static_cast<CpAsyncCopyNode&>(builder.add_library_node<CpAsyncCopyNode>(block, DebugInfo(), 16));
    EXPECT_EQ(copy.code().value(), "cp_async_copy");
    EXPECT_EQ(copy.bytes(), 16u);
    EXPECT_EQ(copy.inputs().size(), 2u); // {_dst, _src}
    EXPECT_EQ(copy.outputs().size(), 0u);

    auto& commit = static_cast<PipelineCommitNode&>(builder.add_library_node<PipelineCommitNode>(block, DebugInfo()));
    EXPECT_EQ(commit.code().value(), "pipeline_commit");

    auto& wait = static_cast<PipelineWaitNode&>(builder.add_library_node<PipelineWaitNode>(block, DebugInfo(), 1));
    EXPECT_EQ(wait.code().value(), "pipeline_wait");
    EXPECT_EQ(wait.keep_outstanding(), 1u);
}

TEST(AsyncCopyNodeTest, Clone) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    auto& copy = static_cast<CpAsyncCopyNode&>(builder.add_library_node<CpAsyncCopyNode>(block, DebugInfo(), 8));
    auto cloned = copy.clone(copy.element_id(), copy.vertex(), copy.get_parent());
    auto* copy_clone = dynamic_cast<CpAsyncCopyNode*>(cloned.get());
    ASSERT_NE(copy_clone, nullptr);
    EXPECT_EQ(copy_clone->bytes(), 8u);

    auto& wait = static_cast<PipelineWaitNode&>(builder.add_library_node<PipelineWaitNode>(block, DebugInfo(), 3));
    auto wcloned = wait.clone(wait.element_id(), wait.vertex(), wait.get_parent());
    auto* wait_clone = dynamic_cast<PipelineWaitNode*>(wcloned.get());
    ASSERT_NE(wait_clone, nullptr);
    EXPECT_EQ(wait_clone->keep_outstanding(), 3u);
}

TEST(AsyncCopyNodeTest, SerializeRoundTrip) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    builder.add_library_node<CpAsyncCopyNode>(block, DebugInfo(), 16);
    builder.add_library_node<PipelineCommitNode>(block, DebugInfo());
    builder.add_library_node<PipelineWaitNode>(block, DebugInfo(), 2);

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(builder.subject());
    auto restored = serializer.deserialize(j);

    auto& rblock = static_cast<structured_control_flow::Block&>(restored->root().at(0));
    size_t n_copy = 0, n_commit = 0, n_wait = 0;
    for (auto& node : rblock.dataflow().nodes()) {
        if (auto* c = dynamic_cast<CpAsyncCopyNode*>(&node)) {
            n_copy++;
            EXPECT_EQ(c->bytes(), 16u);
        } else if (dynamic_cast<PipelineCommitNode*>(&node)) {
            n_commit++;
        } else if (auto* w = dynamic_cast<PipelineWaitNode*>(&node)) {
            n_wait++;
            EXPECT_EQ(w->keep_outstanding(), 2u);
        }
    }
    EXPECT_EQ(n_copy, 1u);
    EXPECT_EQ(n_commit, 1u);
    EXPECT_EQ(n_wait, 1u);
}
