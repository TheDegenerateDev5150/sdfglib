import pytest

import torch
import torch.nn as nn

from docc.torch import compile_torch


@pytest.mark.skip("Requires tensor.extract_slice")
def test_reflection_pad_1d():
    class ReflectionPad1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReflectionPad1d((3, 1))
        def forward(self, x: torch.Tensor):
            return self.pad(x)

    model = ReflectionPad1dNet()
    model_ref = ReflectionPad1dNet()
    example_input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires tensor.extract_slice")
def test_reflection_pad_2d():
    class ReflectionPad2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReflectionPad2d((1, 1, 2, 0))
        def forward(self, x: torch.Tensor):
            return self.pad(x)

    model = ReflectionPad2dNet()
    model_ref = ReflectionPad2dNet()
    example_input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires tensor.extract_slice")
def test_reflection_pad_3d():
    class ReflectionPad3dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReflectionPad3d(1)
        def forward(self, x: torch.Tensor):
            return self.pad(x)

    model = ReflectionPad3dNet()
    model_ref = ReflectionPad3dNet()
    example_input = torch.arange(8, dtype=torch.float).reshape(1, 1, 2, 2, 2)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires tensor.extract_slice")
def test_replicateion_pad_1d():
    class ReplicationPad1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReplicationPad1d((3, 1))
        def forward(self, x: torch.Tensor):
            return self.pad(x)

    model = ReplicationPad1dNet()
    model_ref = ReplicationPad1dNet()
    example_input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires tensor.extract_slice")
def test_replicateion_pad_2d():
    class ReplicationPad2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReplicationPad2d((1, 1, 2, 0))
        def forward(self, x: torch.Tensor):
            return self.pad(x)

    model = ReplicationPad2dNet()
    model_ref = ReplicationPad2dNet()
    example_input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

@pytest.mark.skip("Requires tensor.extract_slice")
def test_replicateion_pad_3d():
    class ReplicationPad3dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ReplicationPad3d((3, 3, 6, 6, 1, 1))
        def forward(self, x: torch.Tensor):
            return self.pad(x)

    model = ReplicationPad3dNet()
    model_ref = ReplicationPad3dNet()
    example_input = torch.randn(16, 3, 8, 320, 480)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

def test_zero_pad_1d():
    class ZeroPad1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ZeroPad1d((3, 1))
        def forward(self, x: torch.Tensor):
            return self.pad(x)

    model = ZeroPad1dNet()
    model_ref = ZeroPad1dNet()
    example_input = torch.randn(1, 2, 4)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

def test_zero_pad_2d():
    class ZeroPad2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ZeroPad2d((1, 1, 2, 0))
        def forward(self, x: torch.Tensor):
            return self.pad(x)
    
    model = ZeroPad2dNet()
    model_ref = ZeroPad2dNet()
    example_input = torch.randn(1, 1, 3, 3)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

def test_zero_pad_3d():
    class ZeroPad3dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ZeroPad3d((3, 3, 6, 6, 0, 1))
        def forward(self, x: torch.Tensor):
            return self.pad(x)
    
    model = ZeroPad3dNet()
    model_ref = ZeroPad3dNet()
    example_input = torch.randn(16, 3, 10, 20, 30)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

def test_constant_pad_1d():
    class ConstantPad1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad1d((3, 1), 3.5)
        def forward(self, x: torch.Tensor):
            return self.pad(x)

    model = ConstantPad1dNet()
    model_ref = ConstantPad1dNet()
    example_input = torch.randn(1, 2, 4)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

def test_constant_pad_2d():
    class ConstantPad2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        def forward(self, x: torch.Tensor):
            return self.pad(x)
    
    model = ConstantPad2dNet()
    model_ref = ConstantPad2dNet()
    example_input = torch.randn(1, 2, 2)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

def test_constant_pad_3d():
    class ConstantPad3dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pad = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
        def forward(self, x: torch.Tensor):
            return self.pad(x)
    
    model = ConstantPad3dNet()
    model_ref = ConstantPad3dNet()
    example_input = torch.randn(16, 3, 10, 20, 30)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

# @pytest.mark.skip("Unsupported by torch-mlir")
# def test_circular_pad_1d():
#     class CircularPad1dNet(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.pad = nn.CircularPad1d((3, 1))
#         def forward(self, x: torch.Tensor):
#             return self.pad(x)

#     model = CircularPad1dNet()
#     model_ref = CircularPad1dNet()
#     example_input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)

#     program = torch.compile(model, backend="docc")
#     with torch.no_grad():
#         res = program(example_input)
#         ref = model_ref(example_input)

#     assert torch.allclose(res, ref)

# @pytest.mark.skip("Unsupported by torch-mlir")
# def test_circular_pad_2d():
#     class CircularPad2dNet(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.pad = nn.CircularPad2d((1, 1, 2, 0))
#         def forward(self, x: torch.Tensor):
#             return self.pad(x)

#     model = CircularPad2dNet()
#     model_ref = CircularPad2dNet()
#     example_input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)

#     program = torch.compile(model, backend="docc")
#     with torch.no_grad():
#         res = program(example_input)
#         ref = model_ref(example_input)

#     assert torch.allclose(res, ref)

# @pytest.mark.skip("Unsupported by torch-mlir")
# def test_circular_pad_3d():
#     class CircularPad3dNet(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.pad = nn.CircularPad3d((3, 3, 6, 6, 1, 1))
#         def forward(self, x: torch.Tensor):
#             return self.pad(x)

#     model = CircularPad3dNet()
#     model_ref = CircularPad3dNet()
#     example_input = torch.randn(16, 3, 8, 320, 480)

#     program = torch.compile(model, backend="docc")
#     with torch.no_grad():
#         res = program(example_input)
#         ref = model_ref(example_input)

#     assert torch.allclose(res, ref)