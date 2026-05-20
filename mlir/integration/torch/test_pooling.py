import torch
import torch.nn as nn

from integration.torch.check import check_backend, check_compile

# --- MaxPool2d ---


def test_maxpool2d_compile():
    class MaxPool2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(MaxPool2dNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


def test_maxpool2d_backend():
    class MaxPool2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(MaxPool2dNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


# --- MaxPool2d kernel 3x3, stride 1 ---


def test_maxpool2d_k3s1_compile():
    class MaxPool2dK3S1Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=3, stride=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(MaxPool2dK3S1Net().eval(), torch.randn(1, 8, 32, 32), rtol=1e-4)


def test_maxpool2d_k3s1_backend():
    class MaxPool2dK3S1Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=3, stride=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(MaxPool2dK3S1Net().eval(), torch.randn(1, 8, 32, 32), rtol=1e-4)


# --- MaxPool2d batch > 1 ---


def test_maxpool2d_batch_compile():
    class MaxPool2dBatchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(MaxPool2dBatchNet().eval(), torch.randn(4, 16, 32, 32), rtol=1e-4)


def test_maxpool2d_batch_backend():
    class MaxPool2dBatchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(MaxPool2dBatchNet().eval(), torch.randn(4, 16, 32, 32), rtol=1e-4)


# --- MaxPool2d with padding ---


def test_maxpool2d_padding_compile():
    class MaxPool2dPaddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=4, stride=2, padding=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(MaxPool2dPaddingNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


def test_maxpool2d_padding_backend():
    class MaxPool2dPaddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=4, stride=2, padding=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(MaxPool2dPaddingNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


# --- Global MaxPool2d (AdaptiveMaxPool2d -> 1x1) ---


def test_global_maxpool2d_compile():
    class GlobalMaxPool2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveMaxPool2d(1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(GlobalMaxPool2dNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


def test_global_maxpool2d_backend():
    class GlobalMaxPool2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveMaxPool2d(1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(GlobalMaxPool2dNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


# --- AvgPool2d ---


def test_avgpool2d_compile():
    class AvgPool2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(AvgPool2dNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


def test_avgpool2d_backend():
    class AvgPool2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(AvgPool2dNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


# --- AvgPool2d kernel 3x3, stride 1 ---


def test_avgpool2d_k3s1_compile():
    class AvgPool2dK3S1Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=3, stride=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(AvgPool2dK3S1Net().eval(), torch.randn(1, 8, 32, 32), rtol=1e-4)


def test_avgpool2d_k3s1_backend():
    class AvgPool2dK3S1Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=3, stride=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(AvgPool2dK3S1Net().eval(), torch.randn(1, 8, 32, 32), rtol=1e-4)


# --- AvgPool2d with padding ---


def test_avgpool2d_padding_compile():
    class AvgPool2dPaddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(AvgPool2dPaddingNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


def test_avgpool2d_padding_backend():
    class AvgPool2dPaddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(AvgPool2dPaddingNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


# --- Global AvgPool2d (AdaptiveAvgPool2d -> 1x1) ---


def test_global_avgpool2d_compile():
    class GlobalAvgPool2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(GlobalAvgPool2dNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


def test_global_avgpool2d_backend():
    class GlobalAvgPool2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(GlobalAvgPool2dNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


# --- Sum pooling (AvgPool2d with divisor_override=1) ---


def test_sumpool2d_compile():
    class SumPool2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, divisor_override=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(SumPool2dNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


def test_sumpool2d_backend():
    class SumPool2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, divisor_override=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(SumPool2dNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


# --- Sum pooling kernel 3x3, stride 1 ---


def test_sumpool2d_k3s1_compile():
    class SumPool2dK3S1Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=3, stride=1, divisor_override=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(SumPool2dK3S1Net().eval(), torch.randn(1, 8, 32, 32), rtol=1e-4)


def test_sumpool2d_k3s1_backend():
    class SumPool2dK3S1Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=3, stride=1, divisor_override=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(SumPool2dK3S1Net().eval(), torch.randn(1, 8, 32, 32), rtol=1e-4)


# --- Sum pooling with padding ---


def test_sumpool2d_padding_compile():
    class SumPool2dPaddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=2, divisor_override=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(SumPool2dPaddingNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


def test_sumpool2d_padding_backend():
    class SumPool2dPaddingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=2, divisor_override=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(SumPool2dPaddingNet().eval(), torch.randn(1, 16, 32, 32), rtol=1e-4)


# --- Sum pooling batch > 1 ---


def test_sumpool2d_batch_compile():
    class SumPool2dBatchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, divisor_override=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_compile(SumPool2dBatchNet().eval(), torch.randn(4, 16, 32, 32), rtol=1e-4)


def test_sumpool2d_batch_backend():
    class SumPool2dBatchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, divisor_override=1)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    check_backend(SumPool2dBatchNet().eval(), torch.randn(4, 16, 32, 32), rtol=1e-4)


# --- Chained MaxPool2d after Conv2d ---


def test_conv_maxpool_compile():
    class ConvMaxPoolNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(self.conv(x))

    check_compile(ConvMaxPoolNet().eval(), torch.randn(1, 3, 32, 32), rtol=1e-4)


def test_conv_maxpool_backend():
    class ConvMaxPoolNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(self.conv(x))

    check_backend(ConvMaxPoolNet().eval(), torch.randn(1, 3, 32, 32), rtol=1e-4)
