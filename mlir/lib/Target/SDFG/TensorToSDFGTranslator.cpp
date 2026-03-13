#include "mlir/Target/SDFG/TensorToSDFGTranslator.h"

#include <cstdint>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
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

    auto new_shape = result_tensor_type.getShape();
    if (!in_tensor_info.is_reshape_valid(new_shape)) {
        return expand_op->emitError("Expand reshape is not valid (non-contiguous or mismatched element count)");
    }

    auto out_tensor_info = in_tensor_info.reshape(new_shape);
    translator.tensor_info_map().insert({out_container, out_tensor_info});

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
        .Default([&](Operation* op) {
            return op->emitError("Unknown operation from tensor dialect encountered: ") << op->getName();
        });
}

} // namespace sdfg
} // namespace mlir
