#include "mlir/Target/SDFG/LinalgToSDFGTranslator.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/fill_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/tasklet_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

namespace mlir {
namespace sdfg {

template<typename ElemOp, ::sdfg::data_flow::TaskletCode fp_code, ::sdfg::data_flow::TaskletCode int_code>
LogicalResult translateLinalgElementwiseTaskletOp(SDFGTranslator& translator, ElemOp* add_op) {
    Value input1 = add_op->getInputs()[0];
    Value input2 = add_op->getInputs()[1];
    Value output = add_op->getOutputs()[0];
    Value result = add_op->getResultTensors()[0];

    auto& builder = translator.builder();
    auto input1_container = translator.get_or_create_container(input1);
    auto input2_container = translator.get_or_create_container(input2);
    auto output_container = translator.get_or_create_container(output);
    auto result_container = translator.get_or_create_container(result);

    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    auto tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

    auto element_type = translator.convertType(result_tensor_type.getElementType());
    auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

    ::sdfg::data_flow::TaskletCode code;
    if (::sdfg::types::is_floating_point(sdfg_tensor->primitive_type())) {
        code = fp_code;
    } else {
        code = int_code;
    }

    translator.add_reference(output_container, result_container);

    auto& block = builder.add_block(translator.insertion_point());
    auto& input1_access = builder.add_access(block, input1_container);
    auto& input2_access =
        *(input1_container == input2_container ? &input1_access : &builder.add_access(block, input2_container));
    auto& result_access = builder.add_access(block, result_container);
    auto& libnode = builder.add_library_node<::sdfg::math::tensor::TaskletTensorNode>(
        block,
        ::sdfg::DebugInfo(),
        code,
        std::vector<std::string>({"_out"}),
        std::vector<std::string>({"_in1", "_in2"}),
        sdfg_tensor->shape()
    );
    builder.add_computational_memlet(block, input1_access, libnode, "_in1", {}, *sdfg_tensor);
    builder.add_computational_memlet(block, input2_access, libnode, "_in2", {}, *sdfg_tensor);
    builder.add_computational_memlet(block, libnode, "_out", result_access, {}, *sdfg_tensor);

    return success();
}

LogicalResult translateLinalgOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<linalg::AddOp>([&](linalg::AddOp add_op) {
            return translateLinalgElementwiseTaskletOp<
                linalg::AddOp,
                ::sdfg::data_flow::TaskletCode::fp_add,
                ::sdfg::data_flow::TaskletCode::int_add>(translator, &add_op);
        })
        .Case<linalg::DivOp>([&](linalg::DivOp div_op) {
            return translateLinalgElementwiseTaskletOp<
                linalg::DivOp,
                ::sdfg::data_flow::TaskletCode::fp_div,
                ::sdfg::data_flow::TaskletCode::int_sdiv>(translator, &div_op);
        })
        .Case<linalg::MulOp>([&](linalg::MulOp div_op) {
            return translateLinalgElementwiseTaskletOp<
                linalg::MulOp,
                ::sdfg::data_flow::TaskletCode::fp_mul,
                ::sdfg::data_flow::TaskletCode::int_mul>(translator, &div_op);
        })
        .Case<linalg::SubOp>([&](linalg::SubOp div_op) {
            return translateLinalgElementwiseTaskletOp<
                linalg::SubOp,
                ::sdfg::data_flow::TaskletCode::fp_sub,
                ::sdfg::data_flow::TaskletCode::int_sub>(translator, &div_op);
        })
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

    Value value = op->value();
    Value output = op->output();
    Value result = op->result();

    auto value_container = translator.get_or_create_container(value);
    auto output_container = translator.get_or_create_container(output);
    auto result_container = translator.get_or_create_container(result);

    translator.add_reference(output_container, result_container);

    auto& block = translator.builder().add_block(sequence);

    auto result_type = dyn_cast_or_null<RankedTensorType>(result.getType());
    if (!result_type) {
        return op->emitError("Only ranked tensor result type is supported for now");
    }

    ::sdfg::types::Scalar base_type(translator.convertType(value.getType())->primitive_type());

    auto tensor_info = translator.get_or_create_tensor_info(translator.get_or_create_container(result), result_type);

    auto tensor_type = tensor_info.get_sdfg_tensor(base_type);


    auto& lib_node =
        translator.builder()
            .add_library_node<::sdfg::math::tensor::FillNode>(block, ::sdfg::DebugInfo(), tensor_type->shape());

    if (auto constant_op = dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
        auto& inaccess = translator.builder().add_constant(
            block, translator.convertTypedAttr(constant_op.getValue()), *translator.convertType(constant_op.getType())
        );
        translator.builder().add_computational_memlet(block, inaccess, lib_node, "X", {}, base_type);
    } else {
        auto& in_access = translator.builder().add_access(block, translator.get_or_create_container(value));
        translator.builder().add_computational_memlet(block, in_access, lib_node, "X", {}, base_type);
    }

    auto& out_access = translator.builder().add_access(block, translator.get_or_create_container(result));
    translator.builder().add_computational_memlet(block, lib_node, "Y", out_access, {}, *tensor_type);

    return success();
}

LogicalResult translateLinalgMatmulOp(SDFGTranslator& translator, linalg::MatmulOp* op) {
    auto& sequence = translator.insertion_point();

    auto output = op->getOutputs()[0];
    auto result = op->getResult(0);

    auto output_container = translator.get_or_create_container(output);
    auto result_container = translator.get_or_create_container(result);

    translator.add_reference(output_container, result_container);

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

    ::sdfg::data_flow::AccessNode* lhs_access = &translator.builder().add_access(block, in_container_lhs);
    ::sdfg::data_flow::AccessNode* rhs_access = &translator.builder().add_access(block, in_container_rhs);

    if (in_container_lhs == in_container_rhs) {
        rhs_access = lhs_access;
    } else {
        rhs_access = &translator.builder().add_access(block, in_container_rhs);
    }

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

    translator.builder().add_computational_memlet(block, *lhs_access, libnode, "A", {}, lhs_tensor_type);
    translator.builder().add_computational_memlet(block, *rhs_access, libnode, "B", {}, rhs_tensor_type);

    auto& write_access = translator.builder().add_access(block, out_container);

    translator.builder().add_computational_memlet(block, libnode, "Y", write_access, {}, output_tensor_type);

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

    translator.add_reference(in_container, out_container);

    // Compute and store tensor info for input and output tensors. This will be used for libnode generation later on.
    auto& in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);

    auto out_tensor_info = in_tensor_info.transpose(permutation);
    translator.tensor_info_map().insert({out_container, out_tensor_info});

    return success();
}

} // namespace sdfg
} // namespace mlir
