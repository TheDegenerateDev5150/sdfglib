#include "mlir/Target/SDFG/LinalgToSDFGTranslator.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/fill_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

namespace mlir {
namespace sdfg {


LogicalResult translateLinalgOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<linalg::FillOp>([&](linalg::FillOp fill_op) { return translateLinalgFillOp(translator, &fill_op); })
        .Case<linalg::MatmulOp>([&](linalg::MatmulOp matmul_op) {
            return translateLinalgMatmulOp(translator, &matmul_op);
        })
        .Case<linalg::TransposeOp>([&](linalg::TransposeOp transpose_op) {
            return translateLinalgTransposeOp(translator, &transpose_op);
        })
        .Default([&](Operation* op) { return op->emitError("Unknown operation from linalg dialect encountered"); });
}

LogicalResult translateLinalgFillOp(SDFGTranslator& translator, linalg::FillOp* op) {
    auto& sequence = translator.insertion_point();

    auto& ref_block = translator.builder().add_block(sequence);

    auto& block = translator.builder().add_block(sequence);

    Value value = op->value();
    Value output = op->output();
    Value result = op->result();

    auto in_type = dyn_cast_or_null<RankedTensorType>(value.getType());
    if (!in_type) {
        return op->emitError("Only ranked tensor fill value is supported for now");
    }

    auto tensor_info = translator.get_or_create_tensor_info(translator.get_or_create_container(result), in_type);
    ::sdfg::symbolic::MultiExpression shape;
    for (auto entry : tensor_info.shape()) {
        shape.push_back(::sdfg::symbolic::integer(entry));
    }
    ::sdfg::types::Tensor input_tensor_type(translator.convertType(in_type)->primitive_type(), shape);

    auto& lib_node =
        translator.builder().add_library_node<::sdfg::math::tensor::FillNode>(block, ::sdfg::DebugInfo(), shape);

    if (auto constant_op = dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
        auto& inaccess = translator.builder().add_constant(
            block, translator.convertTypedAttr(constant_op.getValue()), *translator.convertType(constant_op.getType())
        );
        translator.builder().add_computational_memlet(block, inaccess, lib_node, "_in", {}, input_tensor_type);
    } else {
        auto& in_access = translator.builder().add_access(block, translator.get_or_create_container(value));
        translator.builder().add_computational_memlet(block, in_access, lib_node, "_in", {}, input_tensor_type);
    }

    auto& out_access = translator.builder().add_access(block, translator.get_or_create_container(result));
    translator.builder().add_computational_memlet(block, lib_node, "_out", out_access, {}, input_tensor_type);

    auto& out_access_ref = translator.builder().add_access(ref_block, translator.get_or_create_container(output));

    auto& ref_access = translator.builder().add_access(ref_block, translator.get_or_create_container(result));
    translator.builder().add_reference_memlet(ref_block, out_access_ref, ref_access, {}, input_tensor_type);

    return success();
}

LogicalResult translateLinalgMatmulOp(SDFGTranslator& translator, linalg::MatmulOp* op) {
    auto& sequence = translator.insertion_point();

    auto& ref_block = translator.builder().add_block(sequence);

    auto output = op->getOutputs()[0];
    auto result = op->getResult(0);

    auto& ref_access_in = translator.builder().add_access(ref_block, translator.get_or_create_container(output));
    auto& ref_access_out = translator.builder().add_access(ref_block, translator.get_or_create_container(result));

    auto& block = translator.builder().add_block(sequence);

    // For now, only handle 2D matmul with no transposes or broadcasts
    auto lhs_type = dyn_cast_or_null<RankedTensorType>(op->getOperand(0).getType());
    auto rhs_type = dyn_cast_or_null<RankedTensorType>(op->getOperand(1).getType());
    auto output_type = dyn_cast_or_null<RankedTensorType>(op->getResult(0).getType());
    if (!lhs_type || !rhs_type || !output_type || lhs_type.getRank() != 2 || rhs_type.getRank() != 2 ||
        output_type.getRank() != 2) {
        return op->emitError("Only 2D matmul is supported for now");
    }

    auto in_container_lhs = translator.get_or_create_container(op->getOperand(0));
    auto in_container_rhs = translator.get_or_create_container(op->getOperand(1));
    auto out_container = translator.get_or_create_container(op->getResult(0));

    auto& lhs_access = translator.builder().add_access(block, in_container_lhs);
    auto& rhs_access = translator.builder().add_access(block, in_container_rhs);

    auto& tensor_info_lhs = translator.get_or_create_tensor_info(in_container_lhs, lhs_type);
    auto& tensor_info_rhs = translator.get_or_create_tensor_info(in_container_rhs, rhs_type);
    auto& tensor_info_out = translator.get_or_create_tensor_info(out_container, output_type);

    // check if offsets are 0 for all tensors since we don't support partial tensors for now
    if (tensor_info_lhs.offset() != 0 || tensor_info_rhs.offset() != 0 || tensor_info_out.offset() != 0) {
        return op->emitError("Only tensors with 0 offset are supported for now");
    }

    ::sdfg::symbolic::MultiExpression shape_lhs;
    for (auto entry : tensor_info_lhs.shape()) {
        shape_lhs.push_back(::sdfg::symbolic::integer(entry));
    }
    ::sdfg::symbolic::MultiExpression shape_rhs;
    for (auto entry : tensor_info_rhs.shape()) {
        shape_rhs.push_back(::sdfg::symbolic::integer(entry));
    }
    ::sdfg::symbolic::MultiExpression shape_out;
    for (auto entry : tensor_info_out.shape()) {
        shape_out.push_back(::sdfg::symbolic::integer(entry));
    }

    ::sdfg::symbolic::MultiExpression strides_lhs;
    for (auto entry : tensor_info_lhs.strides()) {
        strides_lhs.push_back(::sdfg::symbolic::integer(entry));
    }
    ::sdfg::symbolic::MultiExpression strides_rhs;
    for (auto entry : tensor_info_rhs.strides()) {
        strides_rhs.push_back(::sdfg::symbolic::integer(entry));
    }

    auto& libnode = translator.builder().add_library_node<::sdfg::math::tensor::MatMulNode>(
        block,
        ::sdfg::DebugInfo(),
        shape_lhs,
        shape_rhs,
        strides_lhs,
        strides_rhs,
        /*offset_a=*/::sdfg::symbolic::zero(),
        /*offset_b=*/::sdfg::symbolic::zero()
    );

    auto lhs_primitive_type = translator.convertType(lhs_type)->primitive_type();
    ::sdfg::types::Tensor lhs_tensor_type(lhs_primitive_type, shape_lhs, strides_lhs);
    auto rhs_primitive_type = translator.convertType(rhs_type)->primitive_type();
    ::sdfg::types::Tensor rhs_tensor_type(rhs_primitive_type, shape_rhs, strides_rhs);
    auto output_primitive_type = translator.convertType(output_type)->primitive_type();
    ::sdfg::types::Tensor output_tensor_type(output_primitive_type, shape_out);

    translator.builder().add_computational_memlet(block, lhs_access, libnode, "A", {}, lhs_tensor_type);
    translator.builder().add_computational_memlet(block, rhs_access, libnode, "B", {}, rhs_tensor_type);

    auto& write_access = translator.builder().add_access(block, out_container);

    translator.builder().add_computational_memlet(block, libnode, "Y", write_access, {}, output_tensor_type);

    translator.builder().add_reference_memlet(ref_block, ref_access_in, ref_access_out, {}, output_tensor_type);

    return success();
}

LogicalResult translateLinalgTransposeOp(SDFGTranslator& translator, linalg::TransposeOp* op) {
    auto& sequence = translator.insertion_point();

    auto& block = translator.builder().add_block(sequence);

    Value input = op->getInput();
    Value result = op->getResult()[0];

    // Check that input and output types are ranked tensors
    auto input_tensor_type = dyn_cast_or_null<TensorType>(input.getType());
    auto result_tensor_type = dyn_cast_or_null<TensorType>(result.getType());
    if (!input_tensor_type || !result_tensor_type) {
        return op->emitError("Input and output types must be ranked tensors");
    }

    auto permutation = op->getPermutation();

    auto in_container = translator.get_or_create_container(input);
    auto out_container = translator.get_or_create_container(result);

    auto& in_access = translator.builder().add_access(block, in_container);
    auto& out_access = translator.builder().add_access(block, out_container);

    translator.builder().add_reference_memlet(
        block,
        in_access,
        out_access,
        {},
        ::sdfg::types::Pointer(::sdfg::types::Scalar(::sdfg::types::PrimitiveType::Void))
    );

    // Compute and store tensor info for input and output tensors. This will be used for libnode generation later on.
    auto& in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);

    auto out_tensor_info = in_tensor_info.transpose(permutation);
    translator.tensor_info_map().insert({out_container, out_tensor_info});

    return success();
}

} // namespace sdfg
} // namespace mlir
