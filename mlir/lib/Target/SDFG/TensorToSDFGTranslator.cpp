#include "mlir/Target/SDFG/TensorToSDFGTranslator.h"

#include <cstdint>
#include <llvm-19/llvm/Support/Casting.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>

#include <string>
#include <vector>
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "mlir/Target/SDFG/helper.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

namespace mlir {
namespace sdfg {

LogicalResult translateTensorEmptyOp(SDFGTranslator& translator, tensor::EmptyOp* empty_op) {
    Value result = empty_op->getResult();
    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());

    std::string container = translator.get_or_create_container(result);
    auto tensor_info = translator.get_or_create_tensor_info(container, result_tensor_type);

    auto element_type = translator.convertType(result_tensor_type.getElementType());
    auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

    uint64_t size = 1;
    for (int64_t dim : tensor_info.shape()) {
        size *= dim;
    }
    translator.handle_malloc(
        container, ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(*element_type))
    );

    return success();
}

LogicalResult translateTensorCollapseOp(SDFGTranslator& translator, tensor::CollapseShapeOp* collapse_op) {
    Value input = collapse_op->getSrc();
    Value result = collapse_op->getResult();

    auto input_tensor_type = llvm::dyn_cast<TensorType>(input.getType());
    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    if (!input_tensor_type || !result_tensor_type) {
        return collapse_op->emitError("Input and output types must be ranked tensors");
    }

    auto in_container = translator.get_or_create_container(input);
    auto out_container = translator.get_or_create_container(result);

    translator.add_reference(in_container, out_container);

    auto& in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);
    auto in_element_type = translator.convertType(input.getType());
    auto& in_scalar_type = static_cast<::sdfg::types::Scalar&>(*in_element_type);
    in_container = translator.store_in_c_order(in_container, in_tensor_info, in_scalar_type);
    in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);

    auto new_shape = result_tensor_type.getShape();
    if (!in_tensor_info.is_reshape_valid(new_shape)) {
        return collapse_op->emitError("Collapse reshape is not valid (non-contiguous or mismatched element count)");
    }

    auto out_tensor_info = in_tensor_info.reshape(new_shape);
    translator.tensor_info_map().insert({out_container, out_tensor_info});

    return success();
}

LogicalResult translateTensorExpandOp(SDFGTranslator& translator, tensor::ExpandShapeOp* expand_op) {
    Value input = expand_op->getSrc();
    Value result = expand_op->getResult();

    auto input_tensor_type = llvm::dyn_cast<TensorType>(input.getType());
    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    if (!input_tensor_type || !result_tensor_type) {
        return expand_op->emitError("Input and output types must be ranked tensors");
    }

    auto in_container = translator.get_or_create_container(input);
    auto out_container = translator.get_or_create_container(result);

    translator.add_reference(in_container, out_container);

    auto& in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);
    auto in_element_type = translator.convertType(input.getType());
    auto& in_scalar_type = static_cast<::sdfg::types::Scalar&>(*in_element_type);
    in_container = translator.store_in_c_order(in_container, in_tensor_info, in_scalar_type);
    in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);

    auto new_shape = result_tensor_type.getShape();
    if (!in_tensor_info.is_reshape_valid(new_shape)) {
        return expand_op->emitError("Expand reshape is not valid (non-contiguous or mismatched element count)");
    }

    auto out_tensor_info = in_tensor_info.reshape(new_shape);
    translator.tensor_info_map().insert({out_container, out_tensor_info});

    return success();
}

LogicalResult translateTensorExtractOp(SDFGTranslator& translator, tensor::ExtractOp* extract_op) {
    Value tensor = extract_op->getTensor();
    auto tensor_container = translator.get_or_create_container(tensor);
    auto tensor_type = llvm::dyn_cast<TensorType>(tensor.getType());
    auto& tensor_info = translator.get_or_create_tensor_info(tensor_container, tensor_type);
    auto element_type = translator.convertType(tensor.getType());
    auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

    OperandRange indices = extract_op->getIndices();
    ::sdfg::data_flow::Subset subset;
    subset.reserve(indices.size());
    for (Value index : indices) {
        subset.push_back(::sdfg::symbolic::symbol(translator.get_or_create_container(index)));
    }

    Value result = extract_op->getResult();
    auto result_container = translator.get_or_create_container(result);

    auto& builder = translator.builder();
    auto& block = builder.add_block(translator.insertion_point());
    auto& tensor_access = builder.add_access(block, tensor_container);
    auto& result_access = builder.add_access(block, result_container);
    auto& tasklet = builder.add_tasklet(block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tensor_access, tasklet, "_in", subset, *sdfg_tensor);
    builder.add_computational_memlet(block, tasklet, "_out", result_access, {});

    return success();
}

LogicalResult translateTensorPadOp(SDFGTranslator& translator, tensor::PadOp* pad_op) {
    Value source = pad_op->getSource();
    Value result = pad_op->getResult();

    // Extract padding values
    std::vector<::sdfg::symbolic::Expression> low, high;
    auto static_low = pad_op->getStaticLow();
    auto static_high = pad_op->getStaticHigh();
    low.reserve(static_low.size());
    high.reserve(static_high.size());
    size_t i = 0;
    for (auto val : static_low) {
        if (val == INT64_MIN) {
            if (i >= pad_op->getLow().size()) {
                return pad_op->emitError("Index out of (non-static) low range: ") << i;
            }
            low.push_back(::sdfg::symbolic::symbol(translator.get_or_create_container(pad_op->getLow()[i++])));
        } else {
            low.push_back(::sdfg::symbolic::integer(val));
        }
    }
    i = 0;
    for (auto val : static_high) {
        if (val == INT64_MIN) {
            if (i >= pad_op->getHigh().size()) {
                return pad_op->emitError("Index out of (non-static) high range: ") << i;
            }
            high.push_back(::sdfg::symbolic::symbol(translator.get_or_create_container(pad_op->getHigh()[i++])));
        } else {
            high.push_back(::sdfg::symbolic::integer(val));
        }
    }

    auto& builder = translator.builder();
    auto source_container = translator.get_or_create_container(source);
    auto result_container = translator.get_or_create_container(result);

    auto source_tensor_type = llvm::dyn_cast<TensorType>(source.getType());
    auto& source_tensor_info = translator.get_or_create_tensor_info(source_container, source_tensor_type);
    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    auto& result_tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

    auto source_element_type = translator.convertType(source_tensor_type.getElementType());
    auto source_sdfg_tensor =
        source_tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*source_element_type));
    auto result_element_type = translator.convertType(result_tensor_type.getElementType());
    auto result_sdfg_tensor =
        result_tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*result_element_type));

    // Allocation for padded container
    uint64_t size = 1;
    for (int64_t dim : result_tensor_info.shape()) {
        size *= dim;
    }
    translator.handle_malloc(
        result_container,
        ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(*result_element_type))
    );

    // Create loops
    ::sdfg::structured_control_flow::Sequence* current_seq = &translator.insertion_point();
    std::vector<std::string> indvars;
    ::sdfg::data_flow::Subset result_subset, source_subset;
    ::sdfg::symbolic::Condition copy_condition = ::sdfg::symbolic::__true__();
    for (i = 0; i < result_tensor_info.shape().size(); i++) {
        int64_t dim = result_tensor_info.shape().at(i);
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, ::sdfg::types::Scalar(sdfg_index_type));
        indvars.push_back(indvar_container);
        auto indvar = ::sdfg::symbolic::symbol(indvar_container);
        result_subset.push_back(indvar);
        source_subset.push_back(::sdfg::symbolic::sub(indvar, low.at(i)));
        auto bound = ::sdfg::symbolic::integer(dim);
        auto condition = ::sdfg::symbolic::Lt(indvar, bound);
        auto init = ::sdfg::symbolic::zero();
        auto update = ::sdfg::symbolic::add(indvar, ::sdfg::symbolic::one());

        if (!::sdfg::symbolic::eq(low.at(i), ::sdfg::symbolic::zero()) ||
            !::sdfg::symbolic::eq(high.at(i), ::sdfg::symbolic::zero())) {
            copy_condition = ::sdfg::symbolic::
                And(copy_condition,
                    ::sdfg::symbolic::
                        And(::sdfg::symbolic::Ge(indvar, low.at(i)),
                            ::sdfg::symbolic::Lt(indvar, ::sdfg::symbolic::sub(bound, high.at(i)))));
        }

        auto& map = builder.add_map(
            *current_seq,
            indvar,
            condition,
            init,
            update,
            ::sdfg::structured_control_flow::ScheduleType_Sequential::create()
        );
        current_seq = &map.root();
    }

    // Create if/else
    auto& if_else = builder.add_if_else(*current_seq);
    auto& copy_case = builder.add_case(if_else, ::sdfg::symbolic::Eq(copy_condition, ::sdfg::symbolic::__true__()));
    auto& fill_case = builder.add_case(if_else, ::sdfg::symbolic::Eq(copy_condition, ::sdfg::symbolic::__false__()));

    // Create copy case
    auto& copy_block = builder.add_block(copy_case);
    auto& source_access = builder.add_access(copy_block, source_container);
    auto& copy_result_access = builder.add_access(copy_block, result_container);
    auto& copy_tasklet = builder.add_tasklet(copy_block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(copy_block, source_access, copy_tasklet, "_in", source_subset, *source_sdfg_tensor);
    builder
        .add_computational_memlet(copy_block, copy_tasklet, "_out", copy_result_access, result_subset, *result_sdfg_tensor);

    // Create block arguments
    Region& region = pad_op->getRegion();
    if (region.getBlocks().size() != 1) {
        return pad_op
                   ->emitOpError("Only exactly one block for the region of tensor.pad is currently supported but found "
                   )
               << region.getBlocks().size();
    }
    auto& block = region.getBlocks().front();
    if (block.getNumArguments() != indvars.size()) {
        return pad_op->emitOpError("number of block arguments != number of tensor dimensions: ")
               << block.getNumArguments() << " != " << indvars.size();
    }
    for (i = 0; i < block.getNumArguments(); i++) {
        BlockArgument argument = block.getArgument(i);
        auto argument_container = translator.get_or_create_container(argument);

        auto& fill_block = builder.add_block(fill_case);
        auto& indvar_access = builder.add_access(fill_block, indvars.at(i));
        auto& argument_access = builder.add_access(fill_block, argument_container);
        auto& tasklet = builder.add_tasklet(fill_block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(fill_block, indvar_access, tasklet, "_in", {});
        builder.add_computational_memlet(fill_block, tasklet, "_out", argument_access, {});
    }

    // Translate operations in block until tensor.yield is reached
    translator.enter_sequence(fill_case);
    for (auto& op : block.getOperations()) {
        if (auto yield_op = llvm::dyn_cast_or_null<tensor::YieldOp>(op)) {
            // Create fill case
            auto yield_container = translator.get_or_create_container(yield_op.getValue());
            auto& fill_block = builder.add_block(translator.insertion_point());
            auto& yield_access = builder.add_access(fill_block, yield_container);
            auto& fill_result_access = builder.add_access(fill_block, result_container);
            auto& tasklet = builder.add_tasklet(fill_block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"});
            builder.add_computational_memlet(fill_block, yield_access, tasklet, "_in", {});
            builder.add_computational_memlet(
                fill_block, tasklet, "_out", fill_result_access, result_subset, *result_sdfg_tensor
            );
            break;
        } else {
            if (failed(translateOp(translator, &op))) {
                return failure();
            }
        }
    }
    translator.exit_sequence(fill_case);

    return success();
}

LogicalResult translateTensorOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<tensor::EmptyOp>([&](tensor::EmptyOp empty_op) { return translateTensorEmptyOp(translator, &empty_op); })
        .Case<tensor::CollapseShapeOp>([&](tensor::CollapseShapeOp collapse_op) {
            return translateTensorCollapseOp(translator, &collapse_op);
        })
        .Case<tensor::ExpandShapeOp>([&](tensor::ExpandShapeOp expand_op) {
            return translateTensorExpandOp(translator, &expand_op);
        })
        .Case<tensor::ExtractOp>([&](tensor::ExtractOp extract_op) {
            return translateTensorExtractOp(translator, &extract_op);
        })
        .Case<tensor::PadOp>([&](tensor::PadOp pad_op) { return translateTensorPadOp(translator, &pad_op); })
        .Default([&](Operation* op) {
            return op->emitError("Unknown operation from tensor dialect encountered: ") << op->getName();
        });
}

} // namespace sdfg
} // namespace mlir
