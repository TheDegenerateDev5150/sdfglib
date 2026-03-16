import pytest

import torch
import torch.nn as nn

from docc.torch import compile_torch


def test_cat():
    class CatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.cat((x, x, x), 0)

    model = CatNet()
    model_ref = CatNet()
    example_input = torch.randn(2, 3)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)

def test_transpose():
    class TransposeNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    model = TransposeNet()
    model_ref = TransposeNet()
    example_input = torch.randn(2, 3)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref)
