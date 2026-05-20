import torch
import torch.nn as nn
import pytest

from integration.torch.check import check_backend, check_compile


# --- Basic 2D transpose ---


def test_2d_backend():
    class Transpose2dNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    check_backend(Transpose2dNet().eval(), torch.randn(8, 10), rtol=1e-5)


def test_2d_compile():
    class Transpose2dNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    check_compile(Transpose2dNet().eval(), torch.randn(8, 10), rtol=1e-5)


# --- .T property ---


def test_t_property_compile():
    class TPropertyNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x.T

    check_compile(TPropertyNet().eval(), torch.randn(8, 10), rtol=1e-5)


def test_t_property_backend():
    class TPropertyNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x.T

    check_backend(TPropertyNet().eval(), torch.randn(8, 10), rtol=1e-5)


# --- Square matrix transpose ---


def test_square_compile():
    class SquareTransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    check_compile(SquareTransposeNet().eval(), torch.randn(10, 10), rtol=1e-5)


def test_square_backend():
    class SquareTransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    check_backend(SquareTransposeNet().eval(), torch.randn(10, 10), rtol=1e-5)


# --- 3D transpose (different dim pairs) ---


def test_3d_dim01_compile():
    class Transpose3dDim01Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    check_compile(Transpose3dDim01Net().eval(), torch.randn(4, 8, 10), rtol=1e-5)


def test_3d_dim01_backend():
    class Transpose3dDim01Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 1)

    check_backend(Transpose3dDim01Net().eval(), torch.randn(4, 8, 10), rtol=1e-5)


def test_3d_dim12_compile():
    class Transpose3dDim12Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 1, 2)

    check_compile(Transpose3dDim12Net().eval(), torch.randn(5, 8, 10), rtol=1e-5)


def test_3d_dim12_backend():
    class Transpose3dDim12Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 1, 2)

    check_backend(Transpose3dDim12Net().eval(), torch.randn(5, 8, 10), rtol=1e-5)


def test_3d_dim02_compile():
    class Transpose3dDim02Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 2)

    check_compile(Transpose3dDim02Net().eval(), torch.randn(6, 8, 10), rtol=1e-5)


def test_3d_dim02_backend():
    class Transpose3dDim02Net(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(x, 0, 2)

    check_backend(Transpose3dDim02Net().eval(), torch.randn(6, 8, 10), rtol=1e-5)


# --- Double transpose (should be identity) ---


def test_double_transpose_compile():
    class DoubleTransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(torch.transpose(x, 0, 1), 0, 1)

    check_compile(DoubleTransposeNet().eval(), torch.randn(9, 10), rtol=1e-5)


def test_double_transpose_backend():
    class DoubleTransposeNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.transpose(torch.transpose(x, 0, 1), 0, 1)

    check_backend(DoubleTransposeNet().eval(), torch.randn(9, 10), rtol=1e-5)


# --- .permute() ---


def test_permute_compile():
    class PermuteNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x.permute(2, 0, 1)

    check_compile(PermuteNet().eval(), torch.randn(7, 8, 10), rtol=1e-5)


def test_permute_backend():
    class PermuteNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x.permute(2, 0, 1)

    check_backend(PermuteNet().eval(), torch.randn(7, 8, 10), rtol=1e-5)
