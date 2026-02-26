import torch
import torch.nn as nn

import docc.torch


def test_pytorch():
    class MatmulNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            h1 = torch.matmul(x, x)
            return h1

    model = MatmulNet()
    example_input = torch.randn(10, 10)

    program = torch.compile(model)
    res = program(example_input)

    res_ref = torch.matmul(example_input, example_input)
    assert torch.allclose(res, res_ref)


def test_backend():
    class MatmulNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            h1 = torch.matmul(x, x)
            return h1

    model = MatmulNet()
    example_input = torch.randn(10, 10)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    res = program(example_input)

    ref_program = torch.compile(model)
    res_ref = ref_program(example_input)
    assert torch.allclose(res, res_ref)


def test_compile():
    class MatmulNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            h1 = torch.matmul(x, x)
            return h1

    model = MatmulNet()
    example_input = torch.randn(10, 10)

    program = docc.torch.compile_torch(model, example_input)
    res = program(example_input)

    res_ref = torch.matmul(example_input, example_input)
    assert torch.allclose(res, res_ref)
