import torch
import torch.nn as nn

from integration.torch.check import check_backend, check_compile


# --- Single linear layer (no bias) ---


def test_single_nobias_compile():
    class SingleNoBiasNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_compile(SingleNoBiasNet(10, 5).eval(), torch.randn(8, 10), rtol=1e-5)


def test_single_nobias_backend():
    class SingleNoBiasNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_backend(SingleNoBiasNet(10, 5).eval(), torch.randn(8, 10), rtol=1e-5)


# --- Single linear layer (with bias) ---


def test_single_bias_compile():
    class SingleBiasNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=True)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_compile(SingleBiasNet(10, 5).eval(), torch.randn(8, 10), rtol=1e-5)


def test_single_bias_backend():
    class SingleBiasNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=True)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_backend(SingleBiasNet(10, 5).eval(), torch.randn(8, 10), rtol=1e-5)


# --- Chained linear layers (no bias) ---


def test_chained_nobias_compile():
    class ChainedNoBiasNet(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_features, bias=False)
            self.linear2 = nn.Linear(hidden_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    check_compile(ChainedNoBiasNet(10, 16, 3).eval(), torch.randn(8, 10), rtol=1e-5)


def test_chained_nobias_backend():
    class ChainedNoBiasNet(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_features, bias=False)
            self.linear2 = nn.Linear(hidden_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    check_backend(ChainedNoBiasNet(10, 16, 3).eval(), torch.randn(8, 10), rtol=1e-5)


# --- Chained linear layers (with bias) ---


def test_chained_bias_compile():
    class ChainedBiasNet(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_features, bias=True)
            self.linear2 = nn.Linear(hidden_features, out_features, bias=True)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    check_compile(ChainedBiasNet(10, 16, 3).eval(), torch.randn(8, 10), rtol=1e-5)


def test_chained_bias_backend():
    class ChainedBiasNet(nn.Module):
        def __init__(self, in_features: int, hidden_features: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, hidden_features, bias=True)
            self.linear2 = nn.Linear(hidden_features, out_features, bias=True)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    check_backend(ChainedBiasNet(10, 16, 3).eval(), torch.randn(8, 10), rtol=1e-5)


# --- Non-square dimensions ---


def test_wide_output_compile():
    class WideOutputNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_compile(WideOutputNet(4, 32).eval(), torch.randn(8, 4), rtol=1e-5)


def test_wide_output_backend():
    class WideOutputNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_backend(WideOutputNet(4, 32).eval(), torch.randn(8, 4), rtol=1e-5)


# --- Narrow bottleneck ---


def test_bottleneck_compile():
    class BottleneckNet(nn.Module):
        def __init__(self, in_features: int, bottleneck: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, bottleneck, bias=False)
            self.linear2 = nn.Linear(bottleneck, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    check_compile(BottleneckNet(32, 2, 32).eval(), torch.randn(8, 32), rtol=1e-5)


def test_bottleneck_backend():
    class BottleneckNet(nn.Module):
        def __init__(self, in_features: int, bottleneck: int, out_features: int):
            super().__init__()
            self.linear1 = nn.Linear(in_features, bottleneck, bias=False)
            self.linear2 = nn.Linear(bottleneck, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear2(self.linear1(x))

    check_backend(BottleneckNet(32, 2, 32).eval(), torch.randn(8, 32), rtol=1e-5)


# --- Single sample (batch_size=1) ---


def test_single_sample_compile():
    class SingleSampleNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_compile(SingleSampleNet(10, 5).eval(), torch.randn(1, 10), rtol=1e-5)


def test_single_sample_backend():
    class SingleSampleNet(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_backend(SingleSampleNet(10, 5).eval(), torch.randn(1, 10), rtol=1e-5)


# --- Deep linear stack (3 layers, no bias) ---


def test_deep_stack_compile():
    class DeepStackNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 16, bias=False)
            self.linear2 = nn.Linear(16, 8, bias=False)
            self.linear3 = nn.Linear(8, 3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear3(self.linear2(self.linear1(x)))

    check_compile(DeepStackNet().eval(), torch.randn(8, 10), rtol=1e-5)


def test_deep_stack_backend():
    class DeepStackNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 16, bias=False)
            self.linear2 = nn.Linear(16, 8, bias=False)
            self.linear3 = nn.Linear(8, 3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear3(self.linear2(self.linear1(x)))

    check_backend(DeepStackNet().eval(), torch.randn(8, 10), rtol=1e-5)


# --- Square linear layer (in_features == out_features) ---


def test_square_compile():
    class SquareLinearNet(nn.Module):
        def __init__(self, features: int):
            super().__init__()
            self.linear = nn.Linear(features, features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_compile(SquareLinearNet(16).eval(), torch.randn(8, 16), rtol=1e-5)


def test_square_backend():
    class SquareLinearNet(nn.Module):
        def __init__(self, features: int):
            super().__init__()
            self.linear = nn.Linear(features, features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_backend(SquareLinearNet(16).eval(), torch.randn(8, 16), rtol=1e-5)


# --- Scalar output (out_features=1) ---


def test_scalar_output_compile():
    class ScalarOutputNet(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, 1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_compile(ScalarOutputNet(10).eval(), torch.randn(8, 10), rtol=1e-5)


def test_scalar_output_backend():
    class ScalarOutputNet(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, 1, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    check_backend(ScalarOutputNet(10).eval(), torch.randn(8, 10), rtol=1e-5)
