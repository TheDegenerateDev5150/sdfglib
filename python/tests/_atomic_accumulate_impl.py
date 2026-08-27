"""Backend-agnostic execution scenarios for the ``AtomicAccumulateNode``.

Bound to the concrete CUDA / ROCm backends by the thin suite modules in
``tests/cuda`` and ``tests/rocm`` via :func:`register` (mirroring
``_gpu_offload_dispatcher_impl``).

Every parallel thread atomically accumulates a value into a shared global slot,
so a correct result proves the atomic add is race-free under real parallel
scopes (block- and grid-level). Following the node's contract, the ``_dst``
pointer is pre-offset by a reference memlet and both edges into the atomic node
carry no subset (the value is staged into a scalar first). Two shapes:
  * single slot: every thread adds ``A[i]`` into ``out[0]``  ⇒  ``out[0] == A.sum()``
  * nested grid x block: thread ``(g, i)`` adds ``A[g*M + i]`` into ``out[i]``
    (the G grid blocks contend on the same M slots).
"""

import numpy as np
import pytest

from docc.sdfg import (
    BufferLifecycle,
    DataTransferDirection,
    Pointer,
    PrimitiveType,
    Scalar,
    StructuredSDFGBuilder,
    TargetLevel,
    TaskletCode,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

# Reuse the shared CUDA/ROCm backend descriptors (device storage, offload
# schedule factory, offloading-block adder, compile target, source glob).
from _gpu_offload_dispatcher_impl import CUDA_BACKEND, ROCM_BACKEND  # noqa: F401

FLOAT_BYTES = 4

# The node's implementation_type string per backend.
_IMPL = {"cuda": "CUDA", "rocm": "ROCm"}


def _scaffold(backend, name, N, N_out):
    """Device buffers + H2D/alloc offloading for A (N) and out (N_out).

    Returns the builder, the device pointer type, and an ``off`` helper that
    emits an offloading block for this backend.
    """
    f = Scalar(PrimitiveType.Float)
    host = Pointer(f)
    dev = Pointer(f, backend.storage())
    i32 = Scalar(PrimitiveType.Int32)

    b = StructuredSDFGBuilder(name)
    b.add_container("A", host, is_argument=True)
    b.add_container("out", host, is_argument=True)
    b.add_container("__daisy_dev_A", dev, is_argument=False)
    b.add_container("__daisy_dev_out", dev, is_argument=False)
    b.add_container("i", i32)

    def off(hc, dc, dirn, life, size):
        backend.offload(b, hc, dc, dirn, life, dev, size)

    off(
        "__daisy_dev_A",
        "__daisy_dev_A",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        f"{N} * {FLOAT_BYTES}",
    )
    off(
        "__daisy_dev_out",
        "__daisy_dev_out",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        f"{N_out} * {FLOAT_BYTES}",
    )
    off(
        "A",
        "__daisy_dev_A",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        f"{N} * {FLOAT_BYTES}",
    )
    off(
        "out",
        "__daisy_dev_out",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        f"{N_out} * {FLOAT_BYTES}",
    )
    return b, dev, off


def _teardown(off, N_out):
    off(
        "out",
        "__daisy_dev_out",
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        f"{N_out} * {FLOAT_BYTES}",
    )
    off(
        "__daisy_dev_A",
        "__daisy_dev_A",
        DataTransferDirection.NONE,
        BufferLifecycle.FREE,
        "0",
    )
    off(
        "__daisy_dev_out",
        "__daisy_dev_out",
        DataTransferDirection.NONE,
        BufferLifecycle.FREE,
        "0",
    )


def _emit_atomic(b, dev, fscalar, impl, src_index, slot):
    """Inside the current (innermost) map, emit three sequential blocks:
    stage ``A[src_index]`` into ``val``, offset ``&out[slot]`` into a pointer, then
    atomically accumulate ``val`` into it. Both edges into the atomic are empty.
    """
    # Block 1: val = A[src_index]  (the subset lives on the staging tasklet)
    blk = b.add_block()
    a_acc = b.add_access(blk, "__daisy_dev_A")
    val_w = b.add_access(blk, "val")
    st = b.add_tasklet(blk, TaskletCode.assign, ["_in"], ["_out"])
    b.add_memlet(blk, a_acc, "", st, "_in", src_index, dev)
    b.add_memlet(blk, st, "_out", val_w, "", "", fscalar)

    # Block 2: dref = &out[slot]  (reference memlet does the offset)
    blk = b.add_block()
    out_acc = b.add_access(blk, "__daisy_dev_out")
    dref = b.add_access(blk, "__daisy_dst_ptr")
    b.add_reference_memlet(blk, out_acc, dref, slot, dev)

    # Block 3: atomicAdd(dref, val)  (no subset on either atomic edge)
    blk = b.add_block()
    dref_r = b.add_access(blk, "__daisy_dst_ptr")
    val_r = b.add_access(blk, "val")
    node = b.add_atomic_accumulate(blk, impl)
    b.add_memlet(blk, dref_r, "", node, "_dst", "", dev)
    b.add_memlet(blk, val_r, "", node, "_src", "", fscalar)


def _build_single_slot(backend, N, scope, psize):
    b, dev, off = _scaffold(backend, "atomic_single", N, 1)
    fscalar = Scalar(PrimitiveType.Float)
    b.add_container("val", fscalar)  # per-thread staged contribution
    b.add_container("__daisy_dst_ptr", dev, is_argument=False)
    b.begin_map("i", "0", str(N), "1", backend.schedule(scope, psize))
    _emit_atomic(b, dev, fscalar, _IMPL[backend.name], src_index="i", slot="0")
    b.end_map()
    _teardown(off, 1)
    return b


def _build_nested(backend, G, M):
    """Nested grid(G) x block(M). Every (g, i) thread adds ``A[g*M + i]`` into
    ``out[i]`` — so the G grid blocks all contend on the same M slots (cross-block
    atomics), and ``_dst`` is pre-offset to ``&out[i]`` by a reference memlet.
    """
    b, dev, off = _scaffold(backend, "atomic_nested", G * M, M)
    fscalar = Scalar(PrimitiveType.Float)
    b.add_container("g", Scalar(PrimitiveType.Int32))
    b.add_container("val", fscalar)  # per-thread staged contribution
    b.add_container(
        "__daisy_dst_ptr", dev, is_argument=False
    )  # per-thread offset pointer
    b.begin_map("g", "0", str(G), "1", backend.schedule(TargetLevel.X_GRID, G))
    b.begin_map("i", "0", str(M), "1", backend.schedule(TargetLevel.X_BLOCK, M))
    _emit_atomic(
        b, dev, fscalar, _IMPL[backend.name], src_index=f"g * {M} + i", slot="i"
    )
    b.end_map()
    b.end_map()
    _teardown(off, M)
    return b


def _compile_run(backend, b, tmp_path, name, args):
    sdfg = b.move()
    sdfg.validate()
    output_dir = tmp_path / name
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), backend.target)

    generated = "\n".join(p.read_text() for p in output_dir.rglob(backend.source_glob))
    assert "atomicAdd(" in generated, "atomic add not emitted"

    compiled = CompiledSDFG(lib_path, sdfg)
    compiled(*args)


def register(namespace, backend):
    """Define the atomic-accumulate tests, bound to ``backend``, into ``namespace``."""

    @pytest.mark.parametrize(
        "N,scope,psize",
        [
            (256, TargetLevel.X_BLOCK, 64),
            (1000, TargetLevel.X_BLOCK, 256),
            (256, TargetLevel.X_GRID, 8),
            (1000, TargetLevel.X_GRID, 16),
        ],
        ids=["block64", "block256", "grid8", "grid16"],
    )
    def test_atomic_accumulate_single_slot(N, scope, psize, tmp_path):
        b = _build_single_slot(backend, N, scope, psize)
        rng = np.random.default_rng(0)
        A = rng.standard_normal((N,)).astype(np.float32)
        out = np.zeros((1,), dtype=np.float32)

        _compile_run(backend, b, tmp_path, f"single_{N}_{psize}", (A, out))

        np.testing.assert_allclose(out[0], A.sum(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "G,M",
        [
            (32, 8),
            (100, 7),
        ],
        ids=["32x8", "100x7"],
    )
    def test_atomic_accumulate_nested(G, M, tmp_path):
        b = _build_nested(backend, G, M)
        rng = np.random.default_rng(1)
        A = rng.standard_normal((G * M,)).astype(np.float32)
        out = np.zeros((M,), dtype=np.float32)

        _compile_run(backend, b, tmp_path, f"nested_{G}_{M}", (A, out))

        expected = A.reshape(G, M).astype(np.float64).sum(axis=0)
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    namespace["test_atomic_accumulate_single_slot"] = test_atomic_accumulate_single_slot
    namespace["test_atomic_accumulate_nested"] = test_atomic_accumulate_nested
