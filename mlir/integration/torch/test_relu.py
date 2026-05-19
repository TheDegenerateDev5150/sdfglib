import torch
import torch.nn as nn

from integration.torch.check import check_backend, check_compile


# --- ReLU: 2-D ---


def test_relu_2d_compile():
    class ReLU2dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(ReLU2dCompile().eval(), torch.randn(4, 64))


def test_relu_2d_backend():
    class ReLU2dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(ReLU2dBackend().eval(), torch.randn(4, 64))


# --- ReLU: 4-D (conv-like) ---


def test_relu_4d_compile():
    class ReLU4dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(ReLU4dCompile().eval(), torch.randn(2, 32, 8, 8))


def test_relu_4d_backend():
    class ReLU4dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(ReLU4dBackend().eval(), torch.randn(2, 32, 8, 8))


# --- ReLU: large tensor ---


def test_relu_large_compile():
    class ReLULargeCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(ReLULargeCompile().eval(), torch.randn(16, 256))


def test_relu_large_backend():
    class ReLULargeBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(ReLULargeBackend().eval(), torch.randn(16, 256))


# --- ReLU6 ---


def test_relu6_2d_compile():
    class ReLU62dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU6()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(ReLU62dCompile().eval(), torch.randn(4, 64))


def test_relu6_2d_backend():
    class ReLU62dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU6()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(ReLU62dBackend().eval(), torch.randn(4, 64))


def test_relu6_4d_compile():
    class ReLU64dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU6()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(ReLU64dCompile().eval(), torch.randn(2, 32, 8, 8))


def test_relu6_4d_backend():
    class ReLU64dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU6()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(ReLU64dBackend().eval(), torch.randn(2, 32, 8, 8))


# --- LeakyReLU ---


def test_leaky_relu_default_compile():
    class LeakyReLUDefaultCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.LeakyReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(LeakyReLUDefaultCompile().eval(), torch.randn(4, 64))


def test_leaky_relu_default_backend():
    class LeakyReLUDefaultBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.LeakyReLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(LeakyReLUDefaultBackend().eval(), torch.randn(4, 64))


def test_leaky_relu_slope_compile():
    class LeakyReLUSlopeCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.LeakyReLU(negative_slope=0.2)

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(LeakyReLUSlopeCompile().eval(), torch.randn(4, 64))


def test_leaky_relu_slope_backend():
    class LeakyReLUSlopeBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.LeakyReLU(negative_slope=0.2)

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(LeakyReLUSlopeBackend().eval(), torch.randn(4, 64))


# --- ELU ---


def test_elu_default_compile():
    class ELUDefaultCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(ELUDefaultCompile().eval(), torch.randn(4, 64))


def test_elu_default_backend():
    class ELUDefaultBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(ELUDefaultBackend().eval(), torch.randn(4, 64))


def test_elu_alpha_compile():
    class ELUAlphaCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ELU(alpha=0.5)

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(ELUAlphaCompile().eval(), torch.randn(4, 64))


def test_elu_alpha_backend():
    class ELUAlphaBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ELU(alpha=0.5)

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(ELUAlphaBackend().eval(), torch.randn(4, 64))


# --- GELU ---


def test_gelu_compile():
    class GELUCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(GELUCompile().eval(), torch.randn(4, 64))


def test_gelu_backend():
    class GELUBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(GELUBackend().eval(), torch.randn(4, 64))


def test_gelu_4d_compile():
    class GELU4dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(GELU4dCompile().eval(), torch.randn(2, 32, 8, 8))


def test_gelu_4d_backend():
    class GELU4dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(GELU4dBackend().eval(), torch.randn(2, 32, 8, 8))


# --- SiLU ---


def test_silu_compile():
    class SiLUCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.SiLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_compile(SiLUCompile().eval(), torch.randn(4, 64))


def test_silu_backend():
    class SiLUBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.SiLU()

        def forward(self, x: torch.Tensor):
            return self.act(x)

    check_backend(SiLUBackend().eval(), torch.randn(4, 64))
