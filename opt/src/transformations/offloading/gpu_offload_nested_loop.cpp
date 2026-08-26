#include "sdfg/transformations/offloading/gpu_offload_nested_loop.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <list>

#include <sdfg/analysis/loop_analysis.h>
#include "sdfg/data_flow/access_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace transformations {

namespace {

// Warp/wavefront size of the target, mirroring the offload dispatchers'
// get_warp_size() (cuda::CUDA_WARP_SIZE / rocm::ROCM_WARP_SIZE).
template<typename GPUType>
int64_t gpu_warp_size();

template<>
int64_t gpu_warp_size<cuda::ScheduleType_CUDA_Offload>() {
    return cuda::CUDA_WARP_SIZE;
}

template<>
int64_t gpu_warp_size<rocm::ScheduleType_ROCM_Offload>() {
    return rocm::ROCM_WARP_SIZE;
}

// Whether @p container is accessed in @p root with a single-dimensional subset
// (`container[index]`) somewhere in the reduce body. The GPU offload reduce
// dispatchers stage the accumulator through a pointer and address it as
// `acc[index]`, so they require exactly that indexed form (see
// accumulator_index() in gpu_offload_reduce_dispatcher.cpp). A bare scalar
// accumulator (empty subset), such as the one the pooling library node emits,
// has no such index and cannot be cooperatively combined; folding it into a GPU
// level would make the dispatcher throw, so it must stay sequential.
bool accumulator_is_indexed(structured_control_flow::Sequence& root, const std::string& container) {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&root};
    while (!queue.empty()) {
        auto* current = queue.front();
        queue.pop_front();

        if (auto* block = dynamic_cast<structured_control_flow::Block*>(current)) {
            auto& dfg = block->dataflow();
            for (auto& memlet : dfg.edges()) {
                const auto* src = dynamic_cast<const data_flow::AccessNode*>(&memlet.src());
                const auto* dst = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst());
                const bool accesses_container = (src != nullptr && src->data() == container) ||
                                                (dst != nullptr && dst->data() == container);
                if (accesses_container && memlet.subset().size() == 1) {
                    return true;
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < seq->size(); ++i) {
                queue.push_back(&seq->at(i));
            }
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&loop->root());
        }
    }
    return false;
}

} // namespace

template<typename GPUType>
GPUOffloadNestedLoop<GPUType>::GPUOffloadNestedLoop(
    structured_control_flow::StructuredLoop& loop, gpu::TargetLevel target_level, symbolic::Integer parallel_size
)
    : loop_(loop), target_level_(target_level), parallel_size_(parallel_size) {}


template<typename GPUType>
std::string GPUOffloadNestedLoop<GPUType>::name() const {
    return "GPUOffloadNestedLoop";
}

template<typename GPUType>
bool GPUOffloadNestedLoop<
    GPUType>::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (dynamic_cast<structured_control_flow::Map*>(&loop_) == nullptr &&
        dynamic_cast<structured_control_flow::Reduce*>(&loop_) == nullptr) {
        return false;
    }

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Condition: Check if parent loop exists
    auto parent = loop_analysis.parent_loop(&loop_);
    if (parent == nullptr) {
        return false;
    }

    // Condition: parallel size must be in bounds of the target level
    if (parallel_size_->as_int() <= 0) {
        return false;
    }
    // X grid dimension is limited to 2^31 - 1.
    if (target_level_ == gpu::TargetLevel::X_GRID) {
        constexpr int64_t max_grid_dim_x = 2147483647; // 2^31 - 1
        if (parallel_size_->as_int() > max_grid_dim_x) {
            return false;
        }
    } else if (target_level_ == gpu::TargetLevel::Y_GRID) {
        constexpr int64_t max_grid_dim_y = 65535; // 2^16 - 1
        if (parallel_size_->as_int() > max_grid_dim_y) {
            return false;
        }
    } else if (target_level_ == gpu::TargetLevel::Z_GRID) {
        constexpr int64_t max_grid_dim_z = 65535; // 2^16 - 1
        if (parallel_size_->as_int() > max_grid_dim_z) {
            return false;
        }
    } else if (target_level_ == gpu::TargetLevel::X_BLOCK) {
        // X block dimension is limited to 1024.
        constexpr int64_t max_block_dim_x = 1024;
        if (parallel_size_->as_int() > max_block_dim_x) {
            return false;
        }
    } else if (target_level_ == gpu::TargetLevel::Y_BLOCK) {
        // Y block dimension is limited to 1024.
        constexpr int64_t max_block_dim_y = 1024;
        if (parallel_size_->as_int() > max_block_dim_y) {
            return false;
        }
    } else if (target_level_ == gpu::TargetLevel::Z_BLOCK) {
        // Z block dimension is limited to 1024.
        constexpr int64_t max_block_dim_z = 1024;
        if (parallel_size_->as_int() > max_block_dim_z) {
            return false;
        }
    } else if (target_level_ == gpu::TargetLevel::WARP) {
        // WARP dimension must equal the target's warp/wavefront size.
        if (parallel_size_->as_int() != gpu_warp_size<GPUType>()) {
            return false;
        }
    } else {
        // Unsupported outermost level
        return false;
    }

    // Condition: check target level nesting is applicable with ancestors loop's schedule type
    auto ancestors = loop_analysis.ancestors(&loop_);

    // Collect the GPU target levels (and parallel sizes) of all enclosing GPU loops of the same target.
    std::vector<std::pair<gpu::TargetLevel, int64_t>> ancestor_levels;
    for (auto* ancestor : ancestors) {
        auto* ancestor_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(ancestor);
        if (ancestor_loop == nullptr) {
            continue;
        }
        const auto& sched = ancestor_loop->schedule_type();
        if (sched.value() != GPUType::value()) {
            continue;
        }
        if (sched.properties().find("target_level") == sched.properties().end()) {
            continue;
        }
        ancestor_levels.emplace_back(GPUType::target_level(sched), GPUType::parallel_size(sched)->as_int());
    }

    // Condition: There must be at least one ancestor GPU loop of the same target level.
    if (ancestor_levels.empty()) {
        return false;
    }

    auto has_ancestor_level = [&](gpu::TargetLevel level) {
        for (auto& [l, s] : ancestor_levels) {
            if (l == level) {
                return true;
            }
        }
        return false;
    };

    // Condition: no dimension may be nested within a WARP (WARP must be the innermost level).
    if (has_ancestor_level(gpu::TargetLevel::WARP)) {
        return false;
    }

    // Condition: a block dimension must be nested within its corresponding grid dimension.
    if (target_level_ == gpu::TargetLevel::X_BLOCK && !has_ancestor_level(gpu::TargetLevel::X_GRID)) {
        return false;
    }
    if (target_level_ == gpu::TargetLevel::Y_BLOCK && !has_ancestor_level(gpu::TargetLevel::Y_GRID)) {
        return false;
    }
    if (target_level_ == gpu::TargetLevel::Z_BLOCK && !has_ancestor_level(gpu::TargetLevel::Z_GRID)) {
        return false;
    }

    // Condition: a WARP must be nested within an X_BLOCK.
    if (target_level_ == gpu::TargetLevel::WARP && !has_ancestor_level(gpu::TargetLevel::X_BLOCK)) {
        return false;
    }

    // Condition: the product of all block dimensions (including this one) must not exceed 1024 threads.
    auto is_block_level = [](gpu::TargetLevel level) {
        return level == gpu::TargetLevel::X_BLOCK || level == gpu::TargetLevel::Y_BLOCK ||
               level == gpu::TargetLevel::Z_BLOCK;
    };
    int64_t block_product = 1;
    // A WARP occupies threads within the block, so it counts toward the block thread budget.
    if (is_block_level(target_level_) || target_level_ == gpu::TargetLevel::WARP) {
        block_product *= parallel_size_->as_int();
    }
    for (auto& [level, size] : ancestor_levels) {
        if (is_block_level(level) || level == gpu::TargetLevel::WARP) {
            block_product *= size;
        }
    }
    constexpr int64_t max_threads_per_block = 1024;
    if (block_product > max_threads_per_block) {
        return false;
    }

    // Condition: num threads >= warp size
    if (target_level_ == gpu::TargetLevel::WARP && block_product < gpu_warp_size<GPUType>()) {
        return false;
    }


    // Condition: nested reduction only support min, max, sum, and product operations
    if (auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(&loop_)) {
        for (auto& reduction : reduce->reductions()) {
            switch (reduction.operation) {
                case structured_control_flow::ReductionOperation::Add:
                case structured_control_flow::ReductionOperation::Mul:
                case structured_control_flow::ReductionOperation::Min:
                case structured_control_flow::ReductionOperation::Max:
                    break;
                default:
                    return false;
            }

            // Condition: the accumulator must be an indexed `acc[index]` container. The
            // offload reduce dispatcher stages the accumulator through a pointer and
            // addresses it as acc[index]; a bare scalar accumulator (empty subset, e.g.
            // the pooling library node's per-thread accumulator) has no such index and
            // cannot be cooperatively combined. Leave such reductions sequential.
            if (!accumulator_is_indexed(reduce->root(), reduction.container)) {
                // A bare scalar accumulator is still supported as a thread-private target at
                // block/warp levels (aliased onto the level's partial, then broadcast to
                // every thread after the combine), but not at grid levels: there is no global
                // slot and a cross-block combine is meaningless. Keep grid-level scalar
                // reductions sequential.
                const bool block_or_warp = target_level_ == gpu::TargetLevel::X_BLOCK ||
                                           target_level_ == gpu::TargetLevel::Y_BLOCK ||
                                           target_level_ == gpu::TargetLevel::Z_BLOCK ||
                                           target_level_ == gpu::TargetLevel::WARP;
                const bool is_scalar = builder.subject().type(reduction.container).type_id() == types::TypeID::Scalar;
                if (!is_scalar || !block_or_warp) {
                    return false;
                }
            }
        }
    }

    // Note: arbitrary `init` and `stride` are permitted. The CUDA dispatcher
    // emits `<map.indvar> = init + thread_flat_id * stride`, so the body sees
    // the natural strided value; `num_iterations()` accounts for both when
    // computing the grid geometry.

    // Condition: Parallelizing this loop must not introduce a data race. Folding a new
    // grid dimension distributes this loop's iterations across the new threads and
    // re-runs every unguarded sibling on each of them, with no grid-wide barrier. That
    // races when this loop produces a shared container a sibling consumes (a reduction
    // accumulator -> consumer, e.g. softmax) or when a sibling read-modify-writes a
    // shared container. Such a loop must be parallelized differently or left sequential.
    if (gpu::nested_parallelization_is_unsafe(loop_, analysis_manager)) {
        return false;
    }

    return true;
}

template<typename GPUType>
void GPUOffloadNestedLoop<
    GPUType>::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    auto new_schedule = GPUType::template create<GPUType>(target_level_, parallel_size_);

    builder.update_schedule_type(loop_, new_schedule);
}

template<typename GPUType>
void GPUOffloadNestedLoop<GPUType>::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["parameters"]["target_level"] = to_string(target_level_);
    j["parameters"]["parallel_size"] = serializer::JSONSerializer::expression(parallel_size_);


    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
}

template<typename GPUType>
GPUOffloadNestedLoop<GPUType> GPUOffloadNestedLoop<
    GPUType>::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    // Prefer the embedding-compatible representation (subgraph/parameters),
    // but fall back to legacy fields (loop/block_size) if needed.
    const auto& subgraph = j.at("subgraph");
    const auto& node_desc = subgraph.at("0");
    size_t loop_id = node_desc.at("element_id").get<size_t>();
    auto target_level = gpu::target_level_from_string(node_desc["parameters"]["target_level"].get<std::string>());

    symbolic::Integer parallel_size =
        SymEngine::rcp_static_cast<const SymEngine::Integer>(symbolic::parse(node_desc["parameters"]["parallel_size"]));
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(builder.find_element_by_id(loop_id));
    if (!loop) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " is not a loop.");
    }
    return GPUOffloadNestedLoop<GPUType>(*loop, target_level, parallel_size);
}

// Explicit template instantiations for the supported GPU targets.
template class GPUOffloadNestedLoop<cuda::ScheduleType_CUDA_Offload>;
template class GPUOffloadNestedLoop<rocm::ScheduleType_ROCM_Offload>;

} // namespace transformations
} // namespace sdfg
