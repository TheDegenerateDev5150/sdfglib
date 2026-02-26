#include "mlir/Target/SDFG/LinalgToSDFGTranslator.h"

#include <llvm-19/llvm/Support/Casting.h>
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
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/cmath_node.h"
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

template<typename ElemOp, ::sdfg::math::cmath::CMathFunction function>
LogicalResult translateLinalgElementwiseCMathOp(SDFGTranslator& translator, ElemOp* op) {
    Value input = op->getInputs()[0];
    Value output = op->getOutputs()[0];
    Value result = op->getResultTensors()[0];

    auto& builder = translator.builder();
    auto input_container = translator.get_or_create_container(input);
    auto output_container = translator.get_or_create_container(output);
    auto result_container = translator.get_or_create_container(result);

    auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
    auto tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

    auto element_type = translator.convertType(result_tensor_type.getElementType());
    auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

    translator.add_reference(output_container, result_container);

    auto& block = builder.add_block(translator.insertion_point());
    auto& input_access = builder.add_access(block, input_container);
    auto& result_access = builder.add_access(block, result_container);
    auto& libnode = builder.add_library_node<::sdfg::math::tensor::CMathTensorNode>(
        block,
        ::sdfg::DebugInfo(),
        function,
        std::vector<std::string>({"_out"}),
        std::vector<std::string>({"_in"}),
        sdfg_tensor->shape()
    );
    builder.add_computational_memlet(block, input_access, libnode, "_in", {}, *sdfg_tensor);
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
        .Case<linalg::AbsOp>([&](linalg::AbsOp abs_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::AbsOp,
                ::sdfg::math::cmath::CMathFunction::fabs>(translator, &abs_op);
        })
        .Case<linalg::CeilOp>([&](linalg::CeilOp ceil_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::CeilOp,
                ::sdfg::math::cmath::CMathFunction::ceil>(translator, &ceil_op);
        })
        .Case<linalg::ErfOp>([&](linalg::ErfOp erf_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::ErfOp,
                ::sdfg::math::cmath::CMathFunction::erf>(translator, &erf_op);
        })
        .Case<linalg::ExpOp>([&](linalg::ExpOp exp_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::ExpOp,
                ::sdfg::math::cmath::CMathFunction::exp>(translator, &exp_op);
        })
        .Case<linalg::FloorOp>([&](linalg::FloorOp floor_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::FloorOp,
                ::sdfg::math::cmath::CMathFunction::floor>(translator, &floor_op);
        })
        .Case<linalg::LogOp>([&](linalg::LogOp log_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::LogOp,
                ::sdfg::math::cmath::CMathFunction::log>(translator, &log_op);
        })
        .Case<linalg::MaxOp>([&](linalg::MaxOp max_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::MaxOp,
                ::sdfg::math::cmath::CMathFunction::fmax>(translator, &max_op);
        })
        .Case<linalg::MinOp>([&](linalg::MinOp min_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::MinOp,
                ::sdfg::math::cmath::CMathFunction::fmin>(translator, &min_op);
        })
        .Case<linalg::PowFOp>([&](linalg::PowFOp powf_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::PowFOp,
                ::sdfg::math::cmath::CMathFunction::pow>(translator, &powf_op);
        })
        .Case<linalg::RoundOp>([&](linalg::RoundOp round_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::RoundOp,
                ::sdfg::math::cmath::CMathFunction::round>(translator, &round_op);
        })
        .Case<linalg::SqrtOp>([&](linalg::SqrtOp sqrt_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::SqrtOp,
                ::sdfg::math::cmath::CMathFunction::sqrt>(translator, &sqrt_op);
        })
        .Case<linalg::TanhOp>([&](linalg::TanhOp tanh_op) {
            return translateLinalgElementwiseCMathOp<
                linalg::TanhOp,
                ::sdfg::math::cmath::CMathFunction::tanh>(translator, &tanh_op);
        })
        .Case<linalg::FillOp>([&](linalg::FillOp fill_op) { return translateLinalgFillOp(translator, &fill_op); })
        .Case<linalg::MatmulOp>([&](linalg::MatmulOp matmul_op) {
            return translateLinalgMatmulOp(translator, &matmul_op);
        })
        .Case<linalg::TransposeOp>([&](linalg::TransposeOp transpose_op) {
            return translateLinalgTransposeOp(translator, &transpose_op);
        })
        .Case<linalg::Conv2DNchwFchwOp>([&](linalg::Conv2DNchwFchwOp conv_op) {
            return translateLinalgConv2DNchwFchwOp(translator, &conv_op);
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

LogicalResult translateLinalgConv2DNchwFchwOp(SDFGTranslator& translator, linalg::Conv2DNchwFchwOp* op) {
    auto& sequence = translator.insertion_point();

    // Get operands
    auto input = op->getInputs()[0]; // X: [N, C_in, H, W]
    auto weight = op->getInputs()[1]; // W: [C_out, C_in/group, kH, kW]
    auto output = op->getOutputs()[0]; // accumulator (from fill)
    auto result = op->getResult(0); // Y: [N, C_out, H_out, W_out]

    auto output_container = translator.get_or_create_container(output);
    auto result_container = translator.get_or_create_container(result);
    translator.add_reference(output_container, result_container);

    // Get tensor types
    auto input_type = dyn_cast_or_null<RankedTensorType>(input.getType());
    auto weight_type = dyn_cast_or_null<RankedTensorType>(weight.getType());
    auto result_type = dyn_cast_or_null<RankedTensorType>(result.getType());
    if (!input_type || !weight_type || !result_type) {
        return op->emitError("Only ranked tensor types are supported for conv2d");
    }
    if (input_type.getRank() != 4 || weight_type.getRank() != 4 || result_type.getRank() != 4) {
        return op->emitError("Only 4D (NCHW) conv2d is supported");
    }

    // Get containers
    auto in_container = translator.get_or_create_container(input);
    auto w_container = translator.get_or_create_container(weight);
    auto out_container = translator.get_or_create_container(result);

    // Get tensor info and element type
    auto element_type = translator.convertType(input_type.getElementType());
    auto scalar_type = static_cast<::sdfg::types::Scalar&>(*element_type);

    // Ensure inputs are in C order (ConvNode requires contiguous data)
    {
        auto& tensor_info_in = translator.get_or_create_tensor_info(in_container, input_type);
        in_container = translator.store_in_c_order(in_container, tensor_info_in, scalar_type);
    }
    {
        auto& tensor_info_w = translator.get_or_create_tensor_info(w_container, weight_type);
        w_container = translator.store_in_c_order(w_container, tensor_info_w, scalar_type);
    }

    // Get tensor info after possible reordering
    auto& final_info_in = translator.get_or_create_tensor_info(in_container, input_type);
    auto& final_info_w = translator.get_or_create_tensor_info(w_container, weight_type);
    auto& tensor_info_out = translator.get_or_create_tensor_info(out_container, result_type);

    // Check offsets are 0
    if (final_info_in.offset() != 0 || final_info_w.offset() != 0 || tensor_info_out.offset() != 0) {
        return op->emitError("Only tensors with 0 offset are supported for conv2d");
    }

    // Build ConvNode parameters

    // Input shape [N, C_in, H, W]
    std::vector<::sdfg::symbolic::Expression> shape;
    for (auto entry : final_info_in.shape()) {
        shape.push_back(::sdfg::symbolic::integer(entry));
    }

    // Kernel shape [kH, kW] from weight dims 2, 3
    std::vector<::sdfg::symbolic::Expression> kernel_shape;
    kernel_shape.push_back(::sdfg::symbolic::integer(final_info_w.shape()[2]));
    kernel_shape.push_back(::sdfg::symbolic::integer(final_info_w.shape()[3]));

    // Strides from op attribute
    std::vector<::sdfg::symbolic::Expression> strides;
    for (int64_t s : op->getStrides().getValues<int64_t>()) {
        strides.push_back(::sdfg::symbolic::integer(s));
    }

    // No padding (padding not supported)
    std::vector<::sdfg::symbolic::Expression> pads = {
        ::sdfg::symbolic::zero(), ::sdfg::symbolic::zero(), ::sdfg::symbolic::zero(), ::sdfg::symbolic::zero()
    };

    // Dilations from op attribute
    std::vector<::sdfg::symbolic::Expression> dilations;
    for (int64_t d : op->getDilations().getValues<int64_t>()) {
        dilations.push_back(::sdfg::symbolic::integer(d));
    }

    // Output channels = weight shape[0] (C_out)
    auto output_channels = ::sdfg::symbolic::integer(final_info_w.shape()[0]);

    // Group = 1 (standard convolution, no groups in NCHW-FCHW)
    auto group = ::sdfg::symbolic::one();

    // Create block and library node
    auto& block = translator.builder().add_block(sequence);
    auto& libnode = translator.builder().add_library_node<::sdfg::math::tensor::ConvNode>(
        block, ::sdfg::DebugInfo(), shape, kernel_shape, strides, pads, dilations, output_channels, group
    );

    // Build tensor types for memlets (all C-order after store_in_c_order)
    auto primitive = scalar_type.primitive_type();

    ::sdfg::symbolic::MultiExpression shape_in;
    for (auto entry : final_info_in.shape()) {
        shape_in.push_back(::sdfg::symbolic::integer(entry));
    }
    ::sdfg::types::Tensor input_tensor_type(primitive, shape_in);

    ::sdfg::symbolic::MultiExpression shape_w;
    for (auto entry : final_info_w.shape()) {
        shape_w.push_back(::sdfg::symbolic::integer(entry));
    }
    ::sdfg::types::Tensor weight_tensor_type(primitive, shape_w);

    ::sdfg::symbolic::MultiExpression shape_out;
    for (auto entry : tensor_info_out.shape()) {
        shape_out.push_back(::sdfg::symbolic::integer(entry));
    }
    ::sdfg::types::Tensor output_tensor_type(primitive, shape_out);

    // Access nodes and memlets
    auto& x_access = translator.builder().add_access(block, in_container);
    auto& w_access = translator.builder().add_access(block, w_container);
    auto& y_access = translator.builder().add_access(block, out_container);

    translator.builder().add_computational_memlet(block, x_access, libnode, "X", {}, input_tensor_type);
    translator.builder().add_computational_memlet(block, w_access, libnode, "W", {}, weight_tensor_type);
    translator.builder().add_computational_memlet(block, libnode, "Y", y_access, {}, output_tensor_type);

    return success();
}

} // namespace sdfg
} // namespace mlir
