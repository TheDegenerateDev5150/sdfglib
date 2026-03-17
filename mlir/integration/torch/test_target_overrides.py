import pytest

import torch
import torch.nn as nn

import docc.torch
from docc.torch import register_target_overrides, register_target
from typing import Callable, Optional, Dict, Any

docc.torch.set_backend_options(target="special", category="server")


def _schedule(sdfg, cat: str, kwargs: Dict[str, any]):
    print("hooking scheduling")
    sdfg.schedule("sequential", cat, False)


def _compile(sdfg, out_dir: str, inst_mode: str, capture: bool, kwargs: Dict[str, Any]):
    print("hooking compile")
    return sdfg._compile(out_dir, "sequential", inst_mode, capture)


def _expand(sdfg, cat: str, kwargs: Dict[str, Any]):
    print("hooking expand")
    return sdfg.expand()


register_target_overrides("special", _schedule, _compile, _expand)


def test_inference(capsys):
    class LinearNet(nn.Module):
        def __init__(self, in_features=4, out_features=2):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = LinearNet()
    model.eval()
    model_ref = LinearNet()
    model_ref.eval()
    model_ref.load_state_dict(model.state_dict())

    program = torch.compile(model, backend="docc")

    example_input = torch.randn(2, 4)

    # Force dynamo (inference) backend
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    captured = capsys.readouterr()
    assert "hooking scheduling" in captured.out
    assert "hooking compile" in captured.out
    assert "hooking expand" in captured.out
    assert "Target 'special' is not supported" not in captured.out

    assert res.shape == (2, 2)
    assert torch.allclose(res, ref, rtol=1e-5)
