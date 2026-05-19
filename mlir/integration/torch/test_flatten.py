import torch
import torch.nn as nn

from integration.torch.check import check_backend, check_compile


# --- Flatten 2D input (batch, features) -> already flat ---


def test_flatten_2d_compile():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=1)

    check_compile(FlatNet().eval(), torch.randn(4, 8))


def test_flatten_2d_backend():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=1)

    check_backend(FlatNet().eval(), torch.randn(4, 8))


# --- Flatten 4D tensor (N, C, H, W) -> (N, C*H*W), as in ResNet-18 ---


def test_flatten_4d_compile():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=1)

    check_compile(FlatNet().eval(), torch.randn(2, 8, 4, 4))


def test_flatten_4d_backend():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=1)

    check_backend(FlatNet().eval(), torch.randn(2, 8, 4, 4))


# --- Flatten 3D tensor (N, S, D) -> (N*S, D) ---


def test_flatten_3d_compile():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=0, end_dim=1)

    check_compile(FlatNet().eval(), torch.randn(3, 5, 16))


def test_flatten_3d_backend():
    class FlatNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x, start_dim=0, end_dim=1)

    check_backend(FlatNet().eval(), torch.randn(3, 5, 16))


# --- Flatten all dims -> 1D ---


def test_flatten_all_compile():
    class FlatAllNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x)

    check_compile(FlatAllNet().eval(), torch.randn(2, 3, 4))


def test_flatten_all_backend():
    class FlatAllNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.flatten(x)

    check_backend(FlatAllNet().eval(), torch.randn(2, 3, 4))


# --- AdaptiveAvgPool2d -> flatten -> Linear (ResNet-18 classifier pattern) ---


def test_flatten_after_avgpool_compile():
    class PoolFlatLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, 10, bias=True)

        def forward(self, x: torch.Tensor):
            x = self.pool(x)  # (N, 512, 1, 1)
            x = torch.flatten(x, start_dim=1)  # (N, 512)
            return self.fc(x)

    check_compile(PoolFlatLinear().eval(), torch.randn(2, 512, 7, 7), rtol=1e-4, atol=1e-5)


def test_flatten_after_avgpool_backend():
    class PoolFlatLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, 10, bias=True)

        def forward(self, x: torch.Tensor):
            x = self.pool(x)  # (N, 512, 1, 1)
            x = torch.flatten(x, start_dim=1)  # (N, 512)
            return self.fc(x)

    check_backend(PoolFlatLinear().eval(), torch.randn(2, 512, 7, 7), rtol=1e-4, atol=1e-5)
