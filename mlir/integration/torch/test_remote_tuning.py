import copy

import torch
import torch.nn as nn

import docc.torch
from docc.torch import TorchProgram


class LinearNet(nn.Module):
    def __init__(self, in_features=8, out_features=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


def test_remote_tuning_init_flag():
    """TorchProgram constructed with remote_tuning=True compiles and gives correct result."""
    model = LinearNet().eval()
    model_ref = copy.deepcopy(model)

    example_input = torch.randn(4, 8)
    program = TorchProgram(model, example_input=example_input, remote_tuning=True)

    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert res.shape == ref.shape
    assert torch.allclose(res, ref, rtol=1e-5)


def test_remote_tuning_compile_override():
    """remote_tuning=True passed directly to compile() compiles and gives correct result."""
    model = LinearNet().eval()
    model_ref = copy.deepcopy(model)

    example_input = torch.randn(4, 8)
    program = TorchProgram(model, example_input=example_input)
    compiled = program.compile(remote_tuning=True)

    with torch.no_grad():
        res = compiled(example_input.numpy())
        ref = model_ref(example_input)

    assert torch.allclose(torch.from_numpy(res) if not isinstance(res, torch.Tensor) else res,
                          ref, rtol=1e-5)


def test_remote_tuning_sequential_target():
    """remote_tuning=True with sequential target compiles and gives correct result."""
    model = LinearNet().eval()
    model_ref = copy.deepcopy(model)

    example_input = torch.randn(4, 8)
    program = TorchProgram(
        model,
        example_input=example_input,
        target="sequential",
        category="server",
        remote_tuning=True,
    )

    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert res.shape == ref.shape
    assert torch.allclose(res, ref, rtol=1e-5)
