import ctypes
import re
import time
from docc.sdfg import Scalar, Array, Pointer, Structure, PrimitiveType

import numpy as np
import ml_dtypes


def idiv(a, b):
    """Integer division (floor division for positive numbers)."""
    return int(a) // int(b)


# Evaluation context for shape expressions
_EVAL_GLOBALS = {"idiv": idiv}

# Pre-compiled regex for _convert_to_python_syntax
_FUNC_CALL_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^()]+)\)")
_PLACEHOLDER_PATTERN = re.compile(
    r"@@@FUNC@@@([a-zA-Z_][a-zA-Z0-9_]*)@@@(.+?)@@@END@@@"
)
_KNOWN_FUNCTIONS = frozenset(
    {"int", "float", "abs", "min", "max", "sum", "len", "idiv"}
)

# Argument type constants for fast dispatch
_ARG_TYPE_OUTPUT_ARRAY = 0
_ARG_TYPE_OUTPUT_SCALAR = 1
_ARG_TYPE_SHAPE = 2
_ARG_TYPE_USER_ARRAY = 3
_ARG_TYPE_USER_STRUCT = 4
_ARG_TYPE_USER_SCALAR = 5

# Pre-cache ctypes.c_int64 for speed
_c_int64 = ctypes.c_int64
_ctypes_cast = ctypes.cast
_ctypes_addressof = ctypes.addressof
_ctypes_byref = ctypes.byref
_ctypes_pointer = ctypes.pointer
_ctypes_c_void_p = ctypes.c_void_p

# Map primitive types to numpy dtypes for fast buffer allocation
_PRIMITIVE_TO_NP_DTYPE = {
    PrimitiveType.Float: np.float32,
    PrimitiveType.Double: np.float64,
    PrimitiveType.Int8: np.int8,
    PrimitiveType.Int16: np.int16,
    PrimitiveType.Int32: np.int32,
    PrimitiveType.Int64: np.int64,
    PrimitiveType.UInt8: np.uint8,
    PrimitiveType.UInt16: np.uint16,
    PrimitiveType.UInt32: np.uint32,
    PrimitiveType.UInt64: np.uint64,
    PrimitiveType.Bool: np.bool_,
    PrimitiveType.Half: np.float16,
    PrimitiveType.BFloat: ml_dtypes.bfloat16,
}

# Pre-computed dtype map for numpy conversion
_NUMPY_DTYPE_MAP = {
    PrimitiveType.Float: np.float32,
    PrimitiveType.Double: np.float64,
    PrimitiveType.Int8: np.int8,
    PrimitiveType.Int16: np.int16,
    PrimitiveType.Int32: np.int32,
    PrimitiveType.Int64: np.int64,
    PrimitiveType.UInt8: np.uint8,
    PrimitiveType.UInt16: np.uint16,
    PrimitiveType.UInt32: np.uint32,
    PrimitiveType.UInt64: np.uint64,
    PrimitiveType.Bool: np.bool_,
    PrimitiveType.Half: np.float16,
    PrimitiveType.BFloat: ml_dtypes.bfloat16,
}

_CTYPES_MAP = {
    PrimitiveType.Bool: ctypes.c_bool,
    PrimitiveType.Int8: ctypes.c_int8,
    PrimitiveType.Int16: ctypes.c_int16,
    PrimitiveType.Int32: ctypes.c_int32,
    PrimitiveType.Int64: ctypes.c_int64,
    PrimitiveType.UInt8: ctypes.c_uint8,
    PrimitiveType.UInt16: ctypes.c_uint16,
    PrimitiveType.UInt32: ctypes.c_uint32,
    PrimitiveType.UInt64: ctypes.c_uint64,
    PrimitiveType.Float: ctypes.c_float,
    PrimitiveType.Double: ctypes.c_double,
    # Half and BFloat are 2 bytes, use c_uint16 for raw storage
    PrimitiveType.Half: ctypes.c_uint16,
    PrimitiveType.BFloat: ctypes.c_uint16,
}


class CompiledSDFG:
    def __init__(
        self,
        lib_path,
        sdfg,
        shape_sources=None,
        structure_member_info=None,
        output_args=None,
        output_shapes=None,
        output_strides=None,
    ):
        self.lib_path = lib_path
        self.sdfg = sdfg
        self.shape_sources = shape_sources or []
        self.structure_member_info = structure_member_info or {}
        self.lib = ctypes.CDLL(lib_path)
        self.func = getattr(self.lib, sdfg.name)

        # Check for output args
        self.output_args = output_args or []
        if not self.output_args and hasattr(sdfg, "metadata"):
            out_args_str = sdfg.metadata("output_args")
            if out_args_str:
                self.output_args = out_args_str.split(",")

        self.output_shapes = output_shapes or {}
        self.output_strides = output_strides or {}

        # Cache for ctypes structure definitions
        self._ctypes_structures = {}

        # Set up argument types
        self.arg_names = sdfg.arguments
        self.arg_types = []
        self.arg_sdfg_types = []  # Keep track of original sdfg types
        for arg_name in sdfg.arguments:
            arg_type = sdfg.type(arg_name)
            self.arg_sdfg_types.append(arg_type)
            ct_type = self._get_ctypes_type(arg_type)
            self.arg_types.append(ct_type)

        self.func.argtypes = self.arg_types

        # Set up return type
        self.func.restype = self._get_ctypes_type(sdfg.return_type)

        # Pre-compute argument classification for fast __call__
        self._precompute_arg_metadata()

    def _precompute_arg_metadata(self):
        """Pre-compute argument metadata for fast __call__ dispatch."""
        output_args_set = set(self.output_args)

        # Build shape source lookup: s_idx -> (u_idx, dim_idx)
        # Also pre-compute the shape keys
        self._shape_sources_list = []  # [(s_idx, u_idx, dim_idx, key_str), ...]
        for i, (u_idx, dim_idx) in enumerate(self.shape_sources):
            self._shape_sources_list.append((i, u_idx, dim_idx, f"_s{i}"))

        # Classify each argument using tuple-based info for faster access
        # Each entry is (arg_type, *type_specific_data)
        self._arg_info = []
        user_arg_counter = 0

        # For output ordering (avoid sorting at runtime)
        output_order = []

        for i, arg_name in enumerate(self.arg_names):
            if arg_name in output_args_set:
                # Output argument
                target_type = self.arg_types[i]
                base_type = target_type._type_
                sdfg_type = self.arg_sdfg_types[i]

                # Pre-compute primitive type for return processing
                primitive_type = None
                if isinstance(sdfg_type, Pointer) and sdfg_type.has_pointee_type():
                    pointee = sdfg_type.pointee_type
                    if isinstance(pointee, Scalar):
                        primitive_type = pointee.primitive_type

                if arg_name in self.output_shapes:
                    dims = self.output_shapes[arg_name]
                    # Always compile shape expressions - they may depend on runtime values
                    compiled_dims = []
                    for d in dims:
                        d_str = str(d)
                        expr = self._convert_to_python_syntax(d_str)
                        compiled_dims.append(compile(expr, "<shape>", "eval"))

                    # Pre-compile stride expressions if available
                    compiled_strides = None
                    if arg_name in self.output_strides:
                        compiled_strides = []
                        for s in self.output_strides[arg_name]:
                            expr = self._convert_to_python_syntax(str(s))
                            compiled_strides.append(compile(expr, "<stride>", "eval"))

                    # Get numpy dtype for fast allocation
                    np_dtype = (
                        _PRIMITIVE_TO_NP_DTYPE.get(primitive_type, np.float64)
                        if primitive_type
                        else np.float64
                    )

                    # Tuple: (arg_type, name, base_type, target_type, compiled_dims, compiled_strides, primitive_type, np_dtype)
                    info_idx = len(self._arg_info)
                    self._arg_info.append(
                        (
                            _ARG_TYPE_OUTPUT_ARRAY,
                            arg_name,
                            base_type,
                            target_type,
                            compiled_dims,
                            compiled_strides,
                            primitive_type,
                            np_dtype,
                        )
                    )
                    output_order.append((int(arg_name.split("_")[-1]), info_idx))
                else:
                    # Scalar return
                    info_idx = len(self._arg_info)
                    self._arg_info.append(
                        (_ARG_TYPE_OUTPUT_SCALAR, arg_name, base_type, primitive_type)
                    )
                    output_order.append((int(arg_name.split("_")[-1]), info_idx))

            elif arg_name.startswith("_s") and arg_name[2:].isdigit():
                # Shape symbol argument - tuple: (arg_type, s_idx, key_str)
                s_idx = int(arg_name[2:])
                self._arg_info.append((_ARG_TYPE_SHAPE, s_idx, f"_s{s_idx}"))
            else:
                # User argument
                sdfg_type = self.arg_sdfg_types[i]
                target_type = self.arg_types[i]
                is_struct_ptr = (
                    sdfg_type
                    and isinstance(sdfg_type, Pointer)
                    and sdfg_type.has_pointee_type()
                    and isinstance(sdfg_type.pointee_type, Structure)
                )

                if is_struct_ptr:
                    struct_name = sdfg_type.pointee_type.name
                    struct_class = self._create_ctypes_structure(struct_name)
                    members = self.structure_member_info[struct_name]
                    sorted_members = tuple(
                        sorted(members.items(), key=lambda x: x[1][0])
                    )
                    # Tuple: (arg_type, user_idx, name, struct_class, sorted_members)
                    self._arg_info.append(
                        (
                            _ARG_TYPE_USER_STRUCT,
                            user_arg_counter,
                            arg_name,
                            struct_class,
                            sorted_members,
                        )
                    )
                elif hasattr(target_type, "contents"):
                    # Array user arg - tuple: (arg_type, user_idx, name, target_type)
                    self._arg_info.append(
                        (_ARG_TYPE_USER_ARRAY, user_arg_counter, arg_name, target_type)
                    )
                else:
                    # Scalar user arg - tuple: (arg_type, user_idx, name, target_type)
                    self._arg_info.append(
                        (_ARG_TYPE_USER_SCALAR, user_arg_counter, arg_name, target_type)
                    )
                user_arg_counter += 1

        self._num_user_args = user_arg_counter

        # Pre-sort output order and build position map
        output_order.sort(key=lambda x: x[0])
        self._output_order = tuple(idx for _, idx in output_order)
        # Map from _arg_info index to result position (for O(1) lookup)
        self._output_pos_map = {idx: pos for pos, idx in enumerate(self._output_order)}

    def _convert_to_python_syntax(self, expr_str):
        result = expr_str

        while True:
            match = _FUNC_CALL_PATTERN.search(result)
            if not match:
                break

            name = match.group(1)
            index = match.group(2)

            if name.lower() in _KNOWN_FUNCTIONS:
                # Use unique delimiters that won't appear in expressions
                placeholder = f"@@@FUNC@@@{name}@@@{index}@@@END@@@"
                result = result[: match.start()] + placeholder + result[match.end() :]
            else:
                result = (
                    result[: match.start()] + f"{name}[{index}]" + result[match.end() :]
                )

        result = _PLACEHOLDER_PATTERN.sub(r"\1(\2)", result)

        return result

    def _create_ctypes_structure(self, struct_name):
        """Create a ctypes Structure class for the given structure name."""
        if struct_name in self._ctypes_structures:
            return self._ctypes_structures[struct_name]

        if struct_name not in self.structure_member_info:
            raise ValueError(f"Structure '{struct_name}' not found in member info")

        # Get member info: {member_name: (index, type)}
        members = self.structure_member_info[struct_name]
        # Sort by index to get correct order
        sorted_members = sorted(members.items(), key=lambda x: x[1][0])

        # Build _fields_ for ctypes.Structure
        fields = []
        for member_name, (index, member_type) in sorted_members:
            ct_type = self._get_ctypes_type(member_type)
            fields.append((member_name, ct_type))

        # Create the ctypes Structure class dynamically
        class CStructure(ctypes.Structure):
            _fields_ = fields

        self._ctypes_structures[struct_name] = CStructure
        return CStructure

    def _get_ctypes_type(self, sdfg_type):
        if isinstance(sdfg_type, Scalar):
            return _CTYPES_MAP.get(sdfg_type.primitive_type, ctypes.c_void_p)
        elif isinstance(sdfg_type, Array):
            # Arrays are passed as pointers
            elem_type = _CTYPES_MAP.get(sdfg_type.primitive_type, ctypes.c_void_p)
            return ctypes.POINTER(elem_type)
        elif isinstance(sdfg_type, Pointer):
            # Check if pointee is a Structure
            # Note: has_pointee_type() is guaranteed to exist on Pointer instances from C++ bindings
            if sdfg_type.has_pointee_type():
                pointee = sdfg_type.pointee_type
                if isinstance(pointee, Structure):
                    # Create ctypes structure and return pointer to it
                    struct_class = self._create_ctypes_structure(pointee.name)
                    return ctypes.POINTER(struct_class)
                elif isinstance(pointee, Scalar):
                    elem_type = _CTYPES_MAP.get(pointee.primitive_type, ctypes.c_void_p)
                    return ctypes.POINTER(elem_type)
            return ctypes.c_void_p
        return ctypes.c_void_p

    def _convert_return_value(self, func_result, shape_symbol_values):
        return_type = self.sdfg.return_type

        if isinstance(return_type, Pointer):
            if return_type.has_pointee_type():
                pointee = return_type.pointee_type
                if isinstance(pointee, Scalar):
                    # Pointer to scalar element type - need to determine array size
                    # Get return shape from metadata if available
                    return_shape_str = self.sdfg.metadata("return_shape")
                    if return_shape_str:
                        # Strip brackets (metadata may be "[10,10]" format)
                        return_shape_str = return_shape_str.strip("[]")
                        shape = []
                        for dim_str in return_shape_str.split(","):
                            try:
                                eval_str = self._convert_to_python_syntax(str(dim_str))
                                val = eval(eval_str, _EVAL_GLOBALS, shape_symbol_values)
                                shape.append(int(val))
                            except Exception:
                                # Can't evaluate shape, return raw pointer
                                return func_result

                        # Determine numpy dtype from primitive type
                        dtype = _NUMPY_DTYPE_MAP.get(pointee.primitive_type, np.float64)

                        # Calculate total size
                        total_size = 1
                        for dim in shape:
                            total_size *= dim

                        # Create numpy array from pointer
                        ct_type = _CTYPES_MAP.get(
                            pointee.primitive_type, ctypes.c_double
                        )
                        arr_type = ct_type * total_size
                        # For Half/BFloat, use np.frombuffer since np.ctypeslib.as_array
                        # doesn't support these types (PEP 3118 buffer format limitation)
                        if pointee.primitive_type in (
                            PrimitiveType.Half,
                            PrimitiveType.BFloat,
                        ):
                            byte_size = total_size * 2  # Half and BFloat are 2 bytes
                            arr = np.frombuffer(
                                (ctypes.c_char * byte_size).from_address(
                                    ctypes.cast(func_result, ctypes.c_void_p).value
                                ),
                                dtype=dtype,
                            ).copy()
                        else:
                            arr = np.ctypeslib.as_array(
                                ctypes.cast(
                                    func_result, ctypes.POINTER(arr_type)
                                ).contents
                            )
                        return arr.reshape(shape)
                    else:
                        # No shape info - try to infer from input shapes
                        # For identity-like operations, the output shape matches input
                        if len(self.shape_sources) > 0 and len(shape_symbol_values) > 0:
                            # Use first input's shape as a fallback
                            shape = []
                            for i in range(len(self.shape_sources)):
                                if f"_s{i}" in shape_symbol_values:
                                    shape.append(shape_symbol_values[f"_s{i}"])

                            if shape:
                                dtype = _NUMPY_DTYPE_MAP.get(
                                    pointee.primitive_type, np.float64
                                )

                                total_size = 1
                                for dim in shape:
                                    total_size *= dim

                                ct_type = _CTYPES_MAP.get(
                                    pointee.primitive_type, ctypes.c_double
                                )
                                arr_type = ct_type * total_size
                                # For Half/BFloat, use np.frombuffer since np.ctypeslib.as_array
                                # doesn't support these types (PEP 3118 buffer format limitation)
                                if pointee.primitive_type in (
                                    PrimitiveType.Half,
                                    PrimitiveType.BFloat,
                                ):
                                    byte_size = (
                                        total_size * 2
                                    )  # Half and BFloat are 2 bytes
                                    arr = np.frombuffer(
                                        (ctypes.c_char * byte_size).from_address(
                                            ctypes.cast(
                                                func_result, ctypes.c_void_p
                                            ).value
                                        ),
                                        dtype=dtype,
                                    ).copy()
                                else:
                                    arr = np.ctypeslib.as_array(
                                        ctypes.cast(
                                            func_result, ctypes.POINTER(arr_type)
                                        ).contents
                                    )
                                return arr.reshape(shape)

                        # Can't determine shape, return raw pointer
                        return func_result
        elif isinstance(return_type, Scalar):
            return func_result

        return func_result

    def __call__(self, *args):
        # Ultra-fast path using pre-computed tuple-based argument info
        # Local variable caching for speed
        _eval = eval
        _GLOBALS = _EVAL_GLOBALS
        _np_empty = np.empty

        # 1. Build shape_symbol_values from shape sources (pre-computed list)
        shape_symbol_values = {}
        for s_idx, u_idx, dim_idx, key in self._shape_sources_list:
            if u_idx < len(args):
                shape_symbol_values[key] = args[u_idx].shape[dim_idx]

        # 2. Process arguments using tuple-based dispatch
        converted_args = []
        structure_refs = (
            []
        )  # Keep refs alive (includes numpy arrays for output buffers)
        return_buffers = (
            []
        )  # List of (np_arr, size, dims, compiled_strides, primitive_type)

        for info in self._arg_info:
            arg_type = info[0]

            if arg_type == _ARG_TYPE_OUTPUT_ARRAY:
                # info = (type, name, base_type, target_type, compiled_dims, compiled_strides, primitive_type, np_dtype)
                target_type = info[3]
                compiled_dims = info[4]
                compiled_strides = info[5]
                np_dtype = info[7]

                # Evaluate size from compiled code objects
                size = 1
                dims = []
                for code in compiled_dims:
                    d = int(_eval(code, _GLOBALS, shape_symbol_values))
                    dims.append(d)
                    size *= d

                # Use numpy for fast allocation (much faster than ctypes)
                buf_arr = _np_empty(size, dtype=np_dtype)
                structure_refs.append(buf_arr)  # Keep alive
                return_buffers.append((buf_arr, size, dims, compiled_strides, info[6]))
                # Get pointer directly from numpy array interface
                ptr = buf_arr.ctypes.data
                converted_args.append(_ctypes_cast(ptr, target_type))

            elif arg_type == _ARG_TYPE_OUTPUT_SCALAR:
                # info = (type, name, base_type, primitive_type)
                base_type = info[2]
                primitive_type = info[3]
                buf = base_type()
                structure_refs.append(buf)
                return_buffers.append((buf, 1, None, None, primitive_type))
                converted_args.append(_ctypes_byref(buf))

            elif arg_type == _ARG_TYPE_SHAPE:
                # info = (type, s_idx, key_str)
                converted_args.append(_c_int64(shape_symbol_values.get(info[2], 0)))

            elif arg_type == _ARG_TYPE_USER_ARRAY:
                # info = (type, user_idx, name, target_type)
                user_idx = info[1]
                arg = args[user_idx]
                shape_symbol_values[info[2]] = arg  # For indirect access
                # Direct pointer access - faster than data_as()
                converted_args.append(_ctypes_cast(arg.ctypes.data, info[3]))

            elif arg_type == _ARG_TYPE_USER_STRUCT:
                # info = (type, user_idx, name, struct_class, sorted_members)
                arg = args[info[1]]
                shape_symbol_values[info[2]] = arg
                struct_class = info[3]
                struct_values = {
                    m[0]: getattr(arg, m[0]) for m in info[4] if hasattr(arg, m[0])
                }
                c_struct = struct_class(**struct_values)
                structure_refs.append(c_struct)
                converted_args.append(_ctypes_pointer(c_struct))

            else:  # _ARG_TYPE_USER_SCALAR
                # info = (type, user_idx, name, target_type)
                arg = args[info[1]]
                shape_symbol_values[info[2]] = arg
                converted_args.append(info[3](arg))

        # 3. Call the function
        func_result = self.func(*converted_args)

        # 4. Process returns using pre-sorted order
        if not return_buffers:
            if func_result is not None:
                return self._convert_return_value(func_result, shape_symbol_values)
            return None

        # return_buffers: [(np_arr_or_ctypes_scalar, size, dims, compiled_strides, primitive_type), ...]
        num_outputs = len(return_buffers)
        results = [None] * num_outputs

        buf_idx = 0
        for i, info in enumerate(self._arg_info):
            arg_type = info[0]
            if arg_type not in (_ARG_TYPE_OUTPUT_ARRAY, _ARG_TYPE_OUTPUT_SCALAR):
                continue

            result_pos = self._output_pos_map[i]
            buf, size, dims, compiled_strides, primitive_type = return_buffers[buf_idx]
            buf_idx += 1

            if arg_type == _ARG_TYPE_OUTPUT_SCALAR:
                # Scalar - buf is a ctypes scalar
                results[result_pos] = buf.value
            else:
                # Array - buf is already a numpy array
                arr = buf
                if dims and len(dims) > 1:
                    # Need to reshape
                    if compiled_strides:
                        try:
                            itemsize = arr.itemsize
                            byte_strides = tuple(
                                int(_eval(s, {}, shape_symbol_values)) * itemsize
                                for s in compiled_strides
                            )
                            arr = np.lib.stride_tricks.as_strided(
                                arr, shape=dims, strides=byte_strides
                            )
                        except:
                            arr = arr.reshape(dims)
                    else:
                        arr = arr.reshape(dims)
                elif dims and len(dims) == 1:
                    pass  # Already 1D with correct size
                results[result_pos] = arr

        if len(results) == 1:
            return results[0]
        return tuple(results) if results else None

    def get_return_shape(self, *args):
        shape_str = self.sdfg.metadata("return_shape")
        if not shape_str:
            return None

        shape_exprs = shape_str.split(",")

        # Reconstruct shape values
        shape_values = {}
        for i, (arg_idx, dim_idx) in enumerate(self.shape_sources):
            arg = args[arg_idx]
            if np is not None and isinstance(arg, np.ndarray):
                val = arg.shape[dim_idx]
                shape_values[f"_s{i}"] = val

        # Add scalar arguments to shape_values
        # We assume the first len(args) arguments in sdfg.arguments correspond to the user arguments
        if hasattr(self.sdfg, "arguments"):
            for arg_name, arg_val in zip(self.sdfg.arguments, args):
                if isinstance(arg_val, (int, np.integer)):
                    shape_values[arg_name] = int(arg_val)

        evaluated_shape = []
        for expr in shape_exprs:
            # Simple evaluation using eval with shape_values
            # Warning: eval is unsafe, but here expressions come from our compiler
            try:
                val = eval(expr, _EVAL_GLOBALS, shape_values)
                evaluated_shape.append(int(val))
            except Exception:
                return None

        return tuple(evaluated_shape)
