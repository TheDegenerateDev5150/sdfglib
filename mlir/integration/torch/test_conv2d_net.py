import torch
import torch.nn as nn

import docc.torch


def test_backend():
    class ConvNet(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, padding=0, bias=False
            )
            self.conv2 = nn.Conv2d(
                hidden_channels, out_channels, kernel_size=3, padding=0, bias=False
            )

        def forward(self, x: torch.Tensor):
            h1 = self.conv1(x)
            h2 = self.conv2(h1)
            return h2

    model = ConvNet(3, 16, 8)
    model_ref = ConvNet(3, 16, 8)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    docc.torch.set_backend_options(target="none", category="server")
    program = torch.compile(model, backend="docc")
    res = program(example_input)

    res_ref = model_ref(example_input)
    assert torch.allclose(res, res_ref, rtol=1e-5)


def test_compile():
    class ConvNet(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, padding=0, bias=False
            )
            self.conv2 = nn.Conv2d(
                hidden_channels, out_channels, kernel_size=3, padding=0, bias=False
            )

        def forward(self, x: torch.Tensor):
            h1 = self.conv1(x)
            h2 = self.conv2(h1)
            return h2

    model = ConvNet(3, 16, 8)
    model_ref = ConvNet(3, 16, 8)
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = docc.torch.compile_torch(model, example_input)
    res = program(example_input)

    res_ref = model_ref(example_input)
    assert torch.allclose(res, res_ref, rtol=1e-5)
