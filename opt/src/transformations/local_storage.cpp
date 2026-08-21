#include "sdfg/transformations/local_storage.h"

#include <functional>
#include <unordered_set>

#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/pointer_analyzers.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace transformations {

namespace {

/// Visit every Block reachable under @p node (recursing through sequences,
/// loops, and if-else branches).
void for_each_block(
    structured_control_flow::ControlFlowNode& node, const std::function<void(structured_control_flow::Block&)>& fn
) {
    if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
        fn(*block);
    } else if (auto* seq = dyn_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); i++) {
            for_each_block(seq->at(i), fn);
        }
    } else if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
        for_each_block(loop->root(), fn);
    } else if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            for_each_block(if_else->at(i).first, fn);
        }
    }
}

/// Escape/overwrite/read/write policy for a single container, fed by the shared
/// pointer analyzers.
struct ContainerAccessPolicy {
    std::string container;
    bool reads = false;
    bool writes = false;
    bool aliased = false; ///< escaped, overwritten, or captured

    void on_escape(const std::string& c, const structured_control_flow::ControlFlowNode*, const Element*) {
        if (c == container) aliased = true;
    }
    void on_overwrite(const std::string& c, const structured_control_flow::ControlFlowNode*, const Element*) {
        if (c == container) aliased = true;
    }
    void on_read_via(const std::string& c, const structured_control_flow::ControlFlowNode*, const data_flow::Memlet*) {
        if (c == container) reads = true;
    }
    void on_write_via(const std::string& c, const structured_control_flow::ControlFlowNode*, const data_flow::Memlet*) {
        if (c == container) writes = true;
    }
};

/// Composes the shared PointerEscape/Overwrite/Used analyzers over a subtree,
/// mirroring MemoryOwnershipAnalysis. Adds one refinement DataDependencyAnalysis
/// carries but the analyzers do not: a library node consuming the pointer with a
/// missing or non-`no_capture` `pointer_access_type` is treated as aliasing.
class ContainerAccessVisitor : public analysis::BaseUserVisitor,
                               public analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>,
                               public analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>,
                               public analysis::PointerUsedAnalyzer<ContainerAccessPolicy> {
    ContainerAccessPolicy& policy_;

    void capture_check(const data_flow::Memlet& edge, const data_flow::DataFlowNode& other) {
        if (auto* lib = dynamic_cast<const data_flow::LibraryNode*>(&other)) {
            auto access = lib->pointer_access_type(edge);
            if (!access || !access->no_capture()) {
                policy_.aliased = true;
            }
        }
    }

public:
    ContainerAccessVisitor(const StructuredSDFG& sdfg, ContainerAccessPolicy& policy)
        : analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>(sdfg, policy),
          analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>(sdfg, policy),
          analysis::PointerUsedAnalyzer<ContainerAccessPolicy>(sdfg, policy), policy_(policy) {}

    void use_as_src_node(
        const std::string& c,
        const data_flow::AccessNode& n,
        const data_flow::Memlet& e,
        const structured_control_flow::Block& b
    ) override {
        analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>::use_as_src_node(c, n, e, b);
        analysis::PointerUsedAnalyzer<ContainerAccessPolicy>::use_as_src_node(c, n, e, b);
        if (c == policy_.container) capture_check(e, e.dst());
    }
    void use_as_dst_node(
        const std::string& c,
        const data_flow::AccessNode& n,
        const data_flow::Memlet& e,
        const structured_control_flow::Block& b
    ) override {
        analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>::use_as_dst_node(c, n, e, b);
        analysis::PointerUsedAnalyzer<ContainerAccessPolicy>::use_as_dst_node(c, n, e, b);
        if (c == policy_.container) capture_check(e, e.src());
    }
    void use_as_return_src(const std::string& c, const structured_control_flow::Return& r) override {
        analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>::use_as_return_src(c, r);
    }
    void use_as_symbol_read(
        const std::string& c,
        const structured_control_flow::ControlFlowNode* n,
        const Element* u,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override {
        analysis::PointerEscapeAnalyzer<ContainerAccessPolicy>::use_as_symbol_read(c, n, u, loc, loc_index, expr);
    }
    void use_as_symbol_write(
        const symbolic::Symbol& c,
        const structured_control_flow::ControlFlowNode* n,
        const Element* u,
        SymbolWriteLocation loc
    ) override {
        analysis::PointerOverwriteAnalyzer<ContainerAccessPolicy>::use_as_symbol_write(c, n, u, loc);
    }
};

} // namespace

LocalStorage::AccessSummary LocalStorage::
    summarize(const StructuredSDFG& sdfg, structured_control_flow::StructuredLoop& loop, const std::string& container) {
    ContainerAccessPolicy policy;
    policy.container = container;
    ContainerAccessVisitor visitor(sdfg, policy);
    visitor.visit(loop.root()); // walks the loop body only
    return AccessSummary{policy.reads, policy.writes, policy.aliased};
}

bool LocalStorage::has_side_effect(structured_control_flow::StructuredLoop& loop) {
    bool found = false;
    for_each_block(loop.root(), [&](structured_control_flow::Block& block) {
        if (found) {
            return;
        }
        for (auto* lib_node : block.dataflow().library_nodes()) {
            if (lib_node->side_effect()) {
                found = true;
                return;
            }
        }
    });
    return found;
}

const analysis::MemoryTileGroup* LocalStorage::tile(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
) {
    auto* groups = analysis_manager.get<analysis::MemoryLayoutAnalysis>().tile_groups(loop, container);
    if (!groups || groups->size() != 1) {
        return nullptr;
    }
    const auto& group = groups->front();
    std::unordered_set<const data_flow::Memlet*> members(group.memlets.begin(), group.memlets.end());

    // Every memlet of the container in the loop body must belong to the group;
    // an unanalyzable (ungrouped) or split memlet makes wholesale rewriting unsafe.
    bool covered = true;
    std::function<void(structured_control_flow::ControlFlowNode&)> walk;
    walk = [&](structured_control_flow::ControlFlowNode& node) {
        if (!covered) {
            return;
        }
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
            auto& dfg = block->dataflow();
            for (auto* access : dfg.data_nodes()) {
                if (access->data() != container) {
                    continue;
                }
                for (auto& memlet : dfg.out_edges(*access)) {
                    if (members.count(&memlet) == 0) {
                        covered = false;
                        return;
                    }
                }
                for (auto& memlet : dfg.in_edges(*access)) {
                    if (members.count(&memlet) == 0) {
                        covered = false;
                        return;
                    }
                }
            }
        } else if (auto* seq = dyn_cast<structured_control_flow::Sequence*>(&node)) {
            for (size_t i = 0; i < seq->size(); i++) {
                walk(seq->at(i));
            }
        } else if (auto* inner = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
            walk(inner->root());
        } else if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&node)) {
            for (size_t i = 0; i < if_else->size(); i++) {
                walk(if_else->at(i).first);
            }
        }
    };
    walk(loop.root());

    return covered ? &group : nullptr;
}

LocalStorage::LocalityPlan LocalStorage::build_locality_plan(
    structured_control_flow::StructuredLoop& loop,
    const TileInfo& tile_info,
    analysis::AnalysisManager& analysis_manager
) {
    LocalityPlan plan;
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    plan.loop_is_outermost = loop_analysis.is_outermost_loop(&loop);

    if (auto* self_map = dynamic_cast<structured_control_flow::Map*>(&loop)) {
        plan.loop_is_gpu = gpu::is_gpu_schedule(self_map->schedule_type());
    }
    for (auto* desc : loop_analysis.descendants(&loop)) {
        auto* m = dynamic_cast<structured_control_flow::Map*>(desc);
        if (m && gpu::is_gpu_schedule(m->schedule_type())) {
            plan.has_gpu_descendant = true;
            break;
        }
    }

    // A dim is cooperative when its induction variable appears in no tile base:
    // every iteration then addresses the same tile and must stage it together.
    auto is_cooperative = [&](const symbolic::Symbol& indvar) {
        for (const auto& base : tile_info.bases) {
            if (symbolic::uses(base, indvar)) {
                return false;
            }
        }
        return true;
    };

    for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop)) {
        auto* map = dynamic_cast<structured_control_flow::Map*>(node);
        if (!map) {
            continue;
        }
        auto& sched = map->schedule_type();
        bool is_gpu = gpu::is_gpu_schedule(sched);
        // Only genuinely parallel loops shape the storage; sequential maps don't.
        if (!is_gpu && sched.category() == structured_control_flow::ScheduleTypeCategory::None) {
            continue;
        }
        LocalityPlan::Dim d;
        d.indvar = map->indvar();
        d.is_gpu = is_gpu;
        d.cooperative = is_cooperative(d.indvar);
        if (is_gpu) {
            const std::string& value = sched.value();
            if (value == "CUDA_Offload" || value == "ROCM_Offload") {
                switch (gpu::gpu_target_level(sched)) {
                    case gpu::TargetLevel::X_GRID:
                    case gpu::TargetLevel::Y_GRID:
                    case gpu::TargetLevel::Z_GRID:
                        d.level = LocalityPlan::Level::Grid;
                        break;
                    case gpu::TargetLevel::X_BLOCK:
                    case gpu::TargetLevel::Y_BLOCK:
                    case gpu::TargetLevel::Z_BLOCK:
                        d.level = LocalityPlan::Level::Block;
                        break;
                    case gpu::TargetLevel::WARP:
                        d.level = LocalityPlan::Level::Warp;
                        break;
                }
                d.parallel_size = gpu::ScheduleType_GPU_Offload::parallel_size(sched);
                d.needs_sync = gpu::ScheduleType_GPU_Offload::nested_sync(sched);
            } else {
                // Legacy CUDA/ROCM: a single fused block-thread dimension.
                d.level = LocalityPlan::Level::Block;
                d.parallel_size = gpu::gpu_block_size(sched);
            }
        }
        plan.dims.push_back(d);
    }
    return plan;
}

LocalStorage::Locality LocalStorage::derive_storage(const LocalityPlan& plan, bool container_written) {
    using Level = LocalityPlan::Level;
    // A cooperative CPU-parallel dim would need threads to share a stack — impossible.
    if (plan.has_cpu_cooperative()) {
        return Locality::Reject;
    }
    if (plan.has_gpu_cooperative()) {
        // Writing a cooperative tile across threads is a reduction the reduce
        // dispatcher cannot lower in device memory.
        if (container_written) {
            return Locality::Reject;
        }
        // A cooperative buffer lives in a device scope inside the kernel, below
        // the outermost loop.
        if (!plan.inside_gpu_kernel() || plan.loop_is_outermost) {
            return Locality::Reject;
        }
        // Storage follows the coarsest cooperative level.
        if (plan.has_cooperative_at(Level::Grid)) {
            return Locality::Global;
        }
        if (plan.has_cooperative_at(Level::Block)) {
            return Locality::Shared;
        }
        // Warp-only cooperation is served by shuffles, not a staged buffer.
        return Locality::Reject;
    }
    // No cooperative dims: a thread-private / sequential buffer. But a host-level
    // loop that is itself GPU-scheduled or wraps a GPU kernel is not a site for
    // a private stack buffer.
    if (!plan.inside_gpu_kernel() && (plan.loop_is_gpu || plan.has_gpu_descendant)) {
        return Locality::Reject;
    }
    return Locality::Private;
}

bool LocalStorage::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    tile_info_ = TileInfo{};
    group_memlets_.clear();
    container_read_ = false;
    container_written_ = false;

    // Container must exist and be a pointer.
    if (!sdfg.exists(container_)) {
        return false;
    }
    if (sdfg.type(container_).type_id() != types::TypeID::Pointer) {
        return false;
    }

    // Classify the container's accesses directly from the dataflow.
    auto summary = summarize(sdfg, loop_, container_);
    container_read_ = summary.reads;
    container_written_ = summary.writes;

    // Aliasing or side effects can reach the container outside the memlets we
    // rewrite, making localization unsound.
    if (summary.aliased) {
        return false;
    }
    if (has_side_effect(loop_)) {
        return false;
    }

    // Nothing to localize unless the container is actually used.
    if (!container_read_ && !container_written_) {
        return false;
    }

    // Resolve the single localizable tile for the whole container.
    auto* group = tile(loop_, container_, analysis_manager);
    if (!group) {
        return false;
    }

    // Extents must be compile-time integer constants.
    if (!is_constant_bounded(group)) {
        return false;
    }

    // Physical capacity: the buffer must fit the target budget.
    auto count = tile_element_count(group);
    if (count.is_null()) {
        return false;
    }
    auto budget = symbolic::integer(static_cast<int64_t>(max_tile_elements()));
    if (!symbolic::is_true(symbolic::Le(count, budget))) {
        return false;
    }

    // Populate tile info + group memlets for apply().
    auto& t = group->tile;
    tile_info_.dimensions = t.extents_approx();
    tile_info_.bases = t.min_subset;
    tile_info_.strides = std::vector<symbolic::Expression>(t.layout.strides().begin(), t.layout.strides().end());
    tile_info_.offset = t.layout.offset();
    group_memlets_.insert(group->memlets.begin(), group->memlets.end());

    // Derive the storage space from the enclosing parallel schedule.
    plan_ = build_locality_plan(loop_, tile_info_, analysis_manager);
    switch (derive_storage(plan_, container_written_)) {
        case Locality::Private:
            storage_type_ = types::StorageType::CPU_Stack();
            break;
        case Locality::Shared: {
            // v1 cooperative path: a single GPU cooperative block dim, read-only,
            // where the cooperative Map is the loop's immediate enclosing loop (so
            // the shared tile is staged exactly once, no re-stage / WAR hazard).
            if (plan_.dims.size() != 1) {
                return false;
            }
            const auto& d = plan_.dims.front();
            if (!d.is_gpu || !d.cooperative || d.level != LocalityPlan::Level::Block) {
                return false;
            }
            if (container_written_) {
                return false;
            }
            // The first loop ancestor must be the cooperative Map itself.
            structured_control_flow::Map* coop_map = nullptr;
            for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop_)) {
                if (auto* enclosing = dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
                    coop_map = dynamic_cast<structured_control_flow::Map*>(enclosing);
                    break;
                }
            }
            if (!coop_map || !symbolic::eq(coop_map->indvar(), d.indvar)) {
                return false;
            }
            // The cooperative copy is lowered by the new offload dispatcher, so
            // only the *_Offload schedules are supported (not the legacy ones).
            const std::string& sched_value = coop_map->schedule_type().value();
            if (sched_value != "CUDA_Offload" && sched_value != "ROCM_Offload") {
                return false;
            }
            storage_type_ = types::StorageType::NV_Shared();
            break;
        }
        case Locality::Global:
            // Grid-cooperative tiles need global memory + grid-wide sync, which is
            // not yet implemented.
            return false;
        case Locality::Reject:
            return false;
    }
    return true;
}

void LocalStorage::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto* parent = dyn_cast<structured_control_flow::Sequence*>(loop_.get_parent());
    if (!parent) {
        throw InvalidTransformationException("LocalStorage: parent of loop must be a Sequence");
    }

    // Element type from a representative group memlet (container type may be opaque).
    auto* representative = *group_memlets_.begin();
    types::Scalar scalar_type(representative->base_type().primitive_type());
    types::Pointer pointer_type(scalar_type);

    local_name_ = builder.find_new_name("__daisy_local_storage_" + container_);

    // Varying dims (extent > 1); extent-1 dims are degenerate and collapsed.
    std::vector<size_t> varying_dims;
    std::vector<symbolic::Expression> varying_dim_sizes;
    for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
        if (!symbolic::eq(tile_info_.dimensions.at(d), symbolic::integer(1))) {
            varying_dims.push_back(d);
            varying_dim_sizes.push_back(tile_info_.dimensions.at(d));
        }
    }

    symbolic::Expression total_size = symbolic::integer(1);
    for (auto& s : varying_dim_sizes) {
        total_size = symbolic::mul(total_size, s);
    }

    types::Array buffer_type(storage_type_, 0, {}, scalar_type, total_size);
    builder.add_container(local_name_, buffer_type);

    // Row-major linearization over the varying dim sizes.
    auto linearize = [&](const std::vector<symbolic::Expression>& indices) -> symbolic::Expression {
        symbolic::Expression linear = symbolic::integer(0);
        symbolic::Expression stride = symbolic::integer(1);
        for (int i = static_cast<int>(indices.size()) - 1; i >= 0; i--) {
            linear = symbolic::add(linear, symbolic::mul(indices[i], stride));
            stride = symbolic::mul(stride, varying_dim_sizes[i]);
        }
        return linear;
    };

    // Original-container linear index for per-varying-dim copy indices.
    auto build_original_subset = [&](const std::vector<symbolic::Expression>& copy_indices) -> data_flow::Subset {
        std::vector<symbolic::Expression> full;
        size_t v = 0;
        for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
            if (!symbolic::eq(tile_info_.dimensions.at(d), symbolic::integer(1))) {
                full.push_back(symbolic::add(tile_info_.bases.at(d), copy_indices.at(v++)));
            } else {
                full.push_back(tile_info_.bases.at(d));
            }
        }
        symbolic::Expression linear = tile_info_.offset;
        for (size_t d = 0; d < full.size(); d++) {
            linear = symbolic::add(linear, symbolic::mul(tile_info_.strides.at(d), full.at(d)));
        }
        return {linear};
    };

    // Emit a nested copy loop nest (one sequential Map per varying dim) either
    // before the loop (copy-in: container -> buffer) or after it (copy-out).
    auto emit_copy = [&](bool writeback) {
        int index = parent->index(loop_) + (writeback ? 1 : 0);
        auto& scope = writeback ? builder.add_sequence_after(*parent, loop_, loop_.debug_info())
                                : builder.add_sequence_before(*parent, loop_, loop_.debug_info());
        structured_control_flow::Sequence* current = &scope;
        std::vector<symbolic::Expression> indvars;
        for (size_t i = 0; i < varying_dims.size(); i++) {
            auto name = builder.find_new_name(
                "__daisy_ls_" + std::string(writeback ? "wb" : "ci") + "_" + container_ + "_d" +
                std::to_string(varying_dims[i])
            );
            builder.add_container(name, types::Scalar(types::PrimitiveType::UInt64));
            auto indvar = symbolic::symbol(name);
            indvars.push_back(indvar);
            auto& map = builder.add_map(
                *current,
                indvar,
                symbolic::Lt(indvar, varying_dim_sizes[i]),
                symbolic::integer(0),
                symbolic::add(indvar, symbolic::integer(1)),
                structured_control_flow::ScheduleType_Sequential::create(),
                loop_.debug_info()
            );
            current = &map.root();
        }

        auto& block = builder.add_block(*current);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto original_subset = build_original_subset(indvars);
        data_flow::Subset buffer_subset = {linearize(indvars)};
        if (writeback) {
            auto& src = builder.add_access(block, local_name_);
            auto& dst = builder.add_access(block, container_);
            builder.add_computational_memlet(block, src, tasklet, "_in", buffer_subset, buffer_type);
            builder.add_computational_memlet(block, tasklet, "_out", dst, original_subset, pointer_type);
        } else {
            auto& src = builder.add_access(block, container_);
            auto& dst = builder.add_access(block, local_name_);
            builder.add_computational_memlet(block, src, tasklet, "_in", original_subset, pointer_type);
            builder.add_computational_memlet(block, tasklet, "_out", dst, buffer_subset, buffer_type);
        }

        builder.move_children(scope, *parent, index + 1);
        builder.remove_child(*parent, index);
    };

    // Cooperative GPU staging: a single flattened Map carrying the cooperative
    // dim's own offload schedule loads the tile into shared memory; the offload
    // dispatcher lowers it to the thread-strided coverage loop.
    auto emit_cooperative_copy_in = [&]() {
        structured_control_flow::Map* coop_map = nullptr;
        for (auto* node : structured_control_flow::ControlFlowNode::parent_chain(loop_)) {
            if (auto* enclosing = dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
                coop_map = dynamic_cast<structured_control_flow::Map*>(enclosing);
                break;
            }
        }

        auto c_name = builder.find_new_name("__daisy_ls_coop_" + container_);
        builder.add_container(c_name, types::Scalar(types::PrimitiveType::UInt64));
        auto c = symbolic::symbol(c_name);

        auto& copy_map = builder.add_map_before(
            *parent,
            loop_,
            c,
            symbolic::Lt(c, total_size),
            symbolic::integer(0),
            symbolic::add(c, symbolic::integer(1)),
            coop_map->schedule_type(),
            loop_.debug_info()
        );

        // Row-major decomposition of the flat index into per-varying-dim indices.
        std::vector<symbolic::Expression> decomp;
        symbolic::Expression remainder = c;
        for (size_t i = 0; i < varying_dim_sizes.size(); i++) {
            if (i + 1 < varying_dim_sizes.size()) {
                symbolic::Expression divisor = symbolic::integer(1);
                for (size_t j = i + 1; j < varying_dim_sizes.size(); j++) {
                    divisor = symbolic::mul(divisor, varying_dim_sizes[j]);
                }
                decomp.push_back(symbolic::div(remainder, divisor));
                remainder = symbolic::mod(remainder, divisor);
            } else {
                decomp.push_back(remainder);
            }
        }

        auto& block = builder.add_block(copy_map.root());
        auto& src = builder.add_access(block, container_);
        auto& dst = builder.add_access(block, local_name_);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, src, tasklet, "_in", build_original_subset(decomp), pointer_type);
        builder.add_computational_memlet(block, tasklet, "_out", dst, {c}, buffer_type);

        // Barrier so every thread's load is visible before the tile is consumed.
        auto& barrier_block = builder.add_block_before(*parent, loop_, loop_.debug_info());
        builder.add_library_node<data_flow::BarrierLocalNode>(barrier_block, DebugInfo());
    };

    if (storage_type_.is_nv_shared()) {
        emit_cooperative_copy_in(); // read-only cooperative tile: copy-in + barrier, no writeback
    } else {
        if (needs_copy_in()) {
            emit_copy(/*writeback=*/false);
        }
        if (needs_copy_out()) {
            emit_copy(/*writeback=*/true);
        }
    }

    // Redirect all container accesses in the loop body to the buffer. v1
    // guarantees single-group full coverage, so every memlet is rewritten and its
    // access node renamed (no split-node handling needed).
    auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();
    std::function<void(structured_control_flow::ControlFlowNode&)> rewrite;
    rewrite = [&](structured_control_flow::ControlFlowNode& node) {
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
            auto& dfg = block->dataflow();
            std::vector<data_flow::AccessNode*> access_nodes;
            for (auto* access_node : dfg.data_nodes()) {
                if (access_node->data() == container_) {
                    access_nodes.push_back(access_node);
                }
            }
            for (auto* access : access_nodes) {
                bool rewrote = false;
                auto rewrite_edge = [&](data_flow::Memlet& memlet) {
                    if (group_memlets_.count(&memlet) == 0) {
                        return;
                    }
                    auto* acc = mla.access(memlet);
                    if (!acc || acc->subset.size() != tile_info_.dimensions.size()) {
                        return;
                    }
                    std::vector<symbolic::Expression> local_indices;
                    for (size_t d = 0; d < tile_info_.dimensions.size(); d++) {
                        if (!symbolic::eq(tile_info_.dimensions.at(d), symbolic::integer(1))) {
                            local_indices.push_back(symbolic::sub(acc->subset.at(d), tile_info_.bases.at(d)));
                        }
                    }
                    memlet.set_subset({linearize(local_indices)});
                    memlet.set_base_type(buffer_type);
                    rewrote = true;
                };
                for (auto& memlet : dfg.out_edges(*access)) {
                    rewrite_edge(memlet);
                }
                for (auto& memlet : dfg.in_edges(*access)) {
                    rewrite_edge(memlet);
                }
                if (rewrote) {
                    access->data(local_name_);
                }
            }
        } else if (auto* seq = dyn_cast<structured_control_flow::Sequence*>(&node)) {
            for (size_t i = 0; i < seq->size(); i++) {
                rewrite(seq->at(i));
            }
        } else if (auto* inner = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
            rewrite(inner->root());
        } else if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&node)) {
            for (size_t i = 0; i < if_else->size(); i++) {
                rewrite(if_else->at(i).first);
            }
        }
    };
    rewrite(loop_.root());

    analysis_manager.invalidate_all();
}

void LocalStorage::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer serializer_full;
    j["parameters"]["storage_type"] = nlohmann::json::object();
    serializer_full.storage_type_to_json(j["parameters"]["storage_type"], storage_type_);

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);

    j["subgraph"]["1"] = nlohmann::json::object();
    j["subgraph"]["1"]["element_id"] = access_node_.element_id();
    j["subgraph"]["1"]["type"] = "access_node";
}

} // namespace transformations
} // namespace sdfg
