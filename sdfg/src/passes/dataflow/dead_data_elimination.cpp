#include "sdfg/passes/dataflow/dead_data_elimination.h"

#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"
#include "sdfg/visualizer/dot_visualizer.h"

namespace sdfg {
namespace passes {

DeadDataElimination::DeadDataElimination() : Pass(), permissive_(false) {};

DeadDataElimination::DeadDataElimination(bool permissive) : Pass(), permissive_(permissive) {};

std::string DeadDataElimination::name() { return "DeadDataElimination"; };

/**
 * Finds memory areas (heap for now) that are wholly owned by the surrounding function. Owned memory can be removed if
 * its no longer used, writes to it can be ellided if the data is never read. This is not true for memory writes in
 * general, as you must prove no reference to that data ever escapes our control
 */
class MemoryOwnershipAnalysis : public visitor::ActualStructuredSDFGVisitor {
    enum class BlockerType { Escape, Overwrite };

    struct OwnedArea {
        data_flow::Memlet* producer;
        symbolic::Expression allocation_size;
        bool non_ssa = false;
    };

private:
    StructuredSDFG& sdfg_;
    // memory that is allocated by us and therefore 'owned' until it escapes
    std::unordered_map<std::string, OwnedArea> originally_owned_data_;
    std::unordered_map<std::string, std::unordered_map<Element*, BlockerType>> blockers_;
    std::unordered_set<std::string> fully_owned_; // never escaped

public:
    MemoryOwnershipAnalysis(StructuredSDFG& sdfg);

    void run(analysis::AnalysisManager& manager);

    bool visit(sdfg::structured_control_flow::Block& node) override;

    bool handleStructuredLoop(StructuredLoop& loop) override;

    bool visit(sdfg::structured_control_flow::Return& node) override;

    const std::unordered_set<std::string>& fully_owned_areas() const { return fully_owned_; }
};

MemoryOwnershipAnalysis::MemoryOwnershipAnalysis(StructuredSDFG& sdfg) : sdfg_(sdfg) {}


void MemoryOwnershipAnalysis::run(analysis::AnalysisManager& manager) {
    dispatch(sdfg_.root());

    for (auto& [name, area] : originally_owned_data_) {
        if (!area.non_ssa && area.allocation_size != SymEngine::null) {
            auto it = blockers_.find(name);
            if (it != blockers_.end()) {
                bool killed = false;
                for (auto& [element, type] : it->second) {
                    if (type != BlockerType::Overwrite || element != area.producer) {
                        killed = true;
                        break;
                    }
                }
                if (!killed) {
                    fully_owned_.insert(name);
                }
            } else {
                fully_owned_.insert(name);
            }
        }
    }
}

bool MemoryOwnershipAnalysis::visit(sdfg::structured_control_flow::Block& node) {
    auto& dflow = node.dataflow();
    for (auto& library_node : dflow.library_nodes()) {
        if (library_node->code() == stdlib::LibraryNodeType_Malloc) {
            auto* malloc_node = dynamic_cast<const stdlib::MallocNode*>(library_node);
            auto& alloc_size = malloc_node->size();
            auto output_conn = malloc_node->output(0);
            auto oedges = dflow.out_edges(*malloc_node) |
                          std::views::filter([&](const auto& e) { return e.src_conn() == output_conn; });
            for (auto& oedge : oedges) {
                auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
                if (access_node && oedge.is_dst_write()) {
                    auto container = access_node->data();
                    auto it = originally_owned_data_.find(container);
                    if (it != originally_owned_data_.end()) {
                        auto& area = it->second;
                        area.non_ssa = true;
                        area.allocation_size = SymEngine::null;
                        area.producer = nullptr;
                        std::cerr << "Conflicting ownership of " << container << std::endl;
                        continue;
                    }
                    originally_owned_data_.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple(container),
                        std::forward_as_tuple(&oedge, alloc_size, false)
                    );
                }
            }
        }
    }

    for (auto& edge : dflow.edges()) {
        auto* access_src_node = dynamic_cast<data_flow::AccessNode*>(&edge.src());
        if ((edge.is_src_pointed_to_address_leak() || edge.is_src_address_leak()) && access_src_node) { // pulls a
                                                                                                        // reference to
                                                                                                        // the owned
                                                                                                        // memory area
                                                                                                        // or can alias
                                                                                                        // the entire
                                                                                                        // pointer
            auto& container = access_src_node->data();
            auto& type = sdfg_.type(container);
            if (type.type_id() == types::TypeID::Pointer) {
                blockers_[container].emplace(&edge, BlockerType::Escape); // it may not be, but this is the safest
                                                                          // assumption
                // other passes can forward the original container and fold it into accesses
            }
        }
        auto* access_dst_node = dynamic_cast<data_flow::AccessNode*>(&edge.dst());
        if (edge.is_dst_write() && access_dst_node) { // writes to the ptr
            auto& container = access_dst_node->data();
            auto& type = sdfg_.type(container);
            if (type.type_id() == types::TypeID::Pointer) {
                blockers_[container].emplace(&edge, BlockerType::Overwrite);
            }
        }
    }
    return true;
}

bool MemoryOwnershipAnalysis::visit(sdfg::structured_control_flow::Return& node) {
    if (node.is_data() && sdfg_.type(node.data()).type_id() == types::TypeID::Pointer) {
        blockers_[node.data()].emplace(&node, BlockerType::Escape);
    }
    return true;
}

bool MemoryOwnershipAnalysis::handleStructuredLoop(StructuredLoop& loop) {
    auto& container = loop.indvar()->get_name();
    if (sdfg_.type(container).type_id() == types::TypeID::Pointer) {
        blockers_[container].emplace(&loop, BlockerType::Overwrite);
    }

    return ActualStructuredSDFGVisitor::handleStructuredLoop(loop);
}

class IndirectMemoryAccessFinder : public visitor::ActualStructuredSDFGVisitor {
private:
    std::unordered_map<std::string, std::unordered_set<data_flow::Memlet*>> indirect_reads_;
    std::unordered_map<std::string, std::unordered_map<data_flow::Memlet*, Block*>> indirect_writes_;
    const std::unordered_set<std::string>& target_containers_;

public:
    IndirectMemoryAccessFinder(const std::unordered_set<std::string>& target_containers);

    const std::unordered_map<std::string, std::unordered_set<data_flow::Memlet*>>& indirect_reads() {
        return indirect_reads_;
    }
    const std::unordered_map<std::string, std::unordered_map<data_flow::Memlet*, Block*>>& indirect_writes() {
        return indirect_writes_;
    }

    bool visit(sdfg::structured_control_flow::Block& node) override;
};

IndirectMemoryAccessFinder::IndirectMemoryAccessFinder(const std::unordered_set<std::string>& target_containers)
    : target_containers_(target_containers) {}

bool IndirectMemoryAccessFinder::visit(sdfg::structured_control_flow::Block& node) {
    auto& dflow = node.dataflow();
    for (auto& access_node : dflow.data_nodes()) {
        auto& container = access_node->data();
        if (target_containers_.contains(container)) {
            for (auto& in_edge : dflow.in_edges(*access_node)) {
                if (in_edge.is_dst_pointed_to_write() || in_edge.is_dst_write()) {
                    indirect_writes_[container][&in_edge] = &node;
                }
            }
            for (auto& out_edge : dflow.out_edges(*access_node)) {
                if (out_edge.is_src_pointed_to_read()) {
                    indirect_reads_[container].insert(&out_edge);
                }
            }
        }
    }
    return true;
}


bool DeadDataElimination::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();

    std::filesystem::path dir = "/home/ramon/.cache/DOCC/test_syrk_intermediate";
    std::filesystem::path p = dir / (sdfg.name() + ".before_dead_data_elimination.dot");
    visualizer::DotVisualizer::writeToFile(sdfg, &p);

    MemoryOwnershipAnalysis ownership_analysis(sdfg);
    ownership_analysis.run(analysis_manager);

    std::unordered_set<std::string> dead_containers;

    auto& fully_owned_areas = ownership_analysis.fully_owned_areas();
    IndirectMemoryAccessFinder remaining_indirects(fully_owned_areas);
    remaining_indirects.dispatch(sdfg.root());
    for (auto& owned_area : fully_owned_areas) {
        auto& all_reads = remaining_indirects.indirect_reads();
        auto cont_reads_it = all_reads.find(owned_area);
        if (cont_reads_it == all_reads.end() || cont_reads_it->second.empty()) {
            DEBUG_PRINTLN("Removing writes of fully owned memory " << owned_area << ", never used!");
            // [owned_area] is never read, no reference to it escapes our control.
            auto& indirect_writes = remaining_indirects.indirect_writes().at(owned_area);
            for (auto& [indirect_write, w_block] : indirect_writes) {
                auto& write_node = dynamic_cast<data_flow::AccessNode&>(indirect_write->dst());
                builder.clear_node(*w_block, write_node);
            }
        }
    }

    auto& users = analysis_manager.get<analysis::Users>();
    auto& data_dependency_analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();

    for (auto& name : sdfg.containers()) {
        if (!sdfg.is_transient(name)) {
            continue;
        }
        if (users.num_views(name) > 0 || users.num_moves(name) > 0) {
            continue;
        }
        auto num_reads = users.num_reads(name);
        if (!num_reads && users.num_writes(name) == 0) {
            dead_containers.insert(name);
            applied = true;
            continue;
        }

        if (sdfg.type(name).type_id() == types::TypeID::Pointer) {
            continue;
            // use analysis does not return actual reads and writes for pointers. So if [name] is a pointer,
            // num reads/writes, does not actually mean no reads exist and any removal is problematic
        }

        bool completely_unused = !num_reads; // if there are reads left, we can never remove the container, but maybe
                                             // some writes
        auto raws = data_dependency_analysis.definitions(name);
        for (auto set : raws) {
            bool no_reads = false;
            if (set.second.size() == 0) {
                no_reads = true;
            }
            if (data_dependency_analysis.is_undefined_user(*set.first)) {
                continue;
            }

            if (no_reads) {
                bool could_eliminate_write = false;
                auto write = set.first;
                if (auto transition = dynamic_cast<structured_control_flow::Transition*>(write->element())) {
                    transition->assignments().erase(symbolic::symbol(name));
                    applied = true;
                    could_eliminate_write = true;
                } else if (auto access_node = dynamic_cast<data_flow::AccessNode*>(write->element())) {
                    auto& graph = access_node->get_parent();

                    auto& src = (*graph.in_edges(*access_node).begin()).src();
                    if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&src)) {
                        auto& block = dynamic_cast<structured_control_flow::Block&>(*graph.get_parent());
                        builder.clear_node(block, *tasklet);
                        applied = true;
                        could_eliminate_write = true;
                    } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(&src)) {
                        if (!library_node->side_effect() ||
                            (permissive_ && library_node->code() == stdlib::LibraryNodeType_Malloc)) {
                            auto& block = dynamic_cast<structured_control_flow::Block&>(*graph.get_parent());
                            builder.clear_node(block, *library_node);
                            applied = true;
                            could_eliminate_write = true;
                        }
                    }
                }

                completely_unused &= could_eliminate_write;
            }
        }

        if (completely_unused) { // no reads, and all remaining writes could be removed
            dead_containers.insert(name);
        }
    }

    for (auto& name : dead_containers) {
        builder.remove_container(name);
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
