import pytest

import torch
import torch.nn as nn

import docc.torch

docc.torch.set_backend_options(target="none", category="server")


def _check_pool(model_cls, model_kwargs, input_shape):
    model = model_cls(**model_kwargs).eval()
    model_ref = model_cls(**model_kwargs).eval()

    model = torch.compile(model, backend="docc")
    model_ref = torch.compile(model_ref)

    x = torch.randn(*input_shape)
    with torch.no_grad():
        output = model(x)
        output_ref = model_ref(x)

    assert output.shape == output_ref.shape
    assert torch.allclose(output, output_ref, rtol=1e-5, atol=1e-5)


def test_maxpool2d():
    class MaxPool2dNet(nn.Module):
        def __init__(self, kernel_size, stride):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        def forward(self, x):
            return self.pool(x)

    _check_pool(
        MaxPool2dNet,
        {"kernel_size": 3, "stride": 2},
        (4, 64, 56, 56),
    )


def test_avgpool2d():
    class AvgPool2dNet(nn.Module):
        def __init__(self, kernel_size, stride):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

        def forward(self, x):
            return self.pool(x)

    _check_pool(
        AvgPool2dNet,
        {"kernel_size": 3, "stride": 2},
        (4, 64, 56, 56),
    )


def test_adaptive_avgpool2d():
    class AdaptiveAvgPool2dNet(nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(output_size=output_size)

        def forward(self, x):
            return self.pool(x)

    _check_pool(
        AdaptiveAvgPool2dNet,
        {"output_size": (7, 7)},
        (4, 512, 28, 28),
    )


@pytest.mark.skip()
def test_placeholder():
    """Placeholder — extend with additional pooling variants as needed."""
    pass
