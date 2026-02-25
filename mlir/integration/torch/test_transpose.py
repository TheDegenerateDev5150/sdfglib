import torch
import torch.nn as nn

import docc.torch


def test_pytorch():
    class TransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            h1 = torch.transpose(x, 0, 1)
            return h1

    model = TransposeNet()
    example_input = torch.randn(8, 10)

    program = torch.compile(model)
    res = program(example_input)

    res_ref = torch.transpose(example_input, 0, 1)
    assert torch.allclose(res, res_ref)


def test_backend():
    class TransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            h1 = torch.transpose(x, 0, 1)
            return h1

    model = TransposeNet()
    example_input = torch.randn(8, 10)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    res = program(example_input)

    res_ref = torch.transpose(example_input, 0, 1)
    assert torch.allclose(res, res_ref)


def test_compile():
    class TransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            h1 = torch.transpose(x, 0, 1)
            return h1

    model = TransposeNet()
    example_input = torch.randn(8, 10)

    program = docc.torch.compile_torch(model, example_input)
    res = program(example_input)

    res_ref = torch.transpose(example_input, 0, 1)
    assert torch.allclose(res, res_ref)
