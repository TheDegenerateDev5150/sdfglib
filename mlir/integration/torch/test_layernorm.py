import torch
import torch.nn as nn

from integration.torch.check import check_backend, check_compile


# --- Normalize last dim ---


def test_layernorm_last_dim_compile():
    class LNLastDimCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(128)

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_compile(LNLastDimCompile().eval(), torch.randn(4, 16, 128))


def test_layernorm_last_dim_backend():
    class LNLastDimBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(128)

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_backend(LNLastDimBackend().eval(), torch.randn(4, 16, 128))


# --- Normalize last two dims ---


def test_layernorm_2d_shape_compile():
    class LN2dShapeCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm([8, 16])

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_compile(LN2dShapeCompile().eval(), torch.randn(4, 8, 16))


def test_layernorm_2d_shape_backend():
    class LN2dShapeBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm([8, 16])

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_backend(LN2dShapeBackend().eval(), torch.randn(4, 8, 16))


# --- Large hidden dim ---


def test_layernorm_large_compile():
    class LNLargeCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(512)

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_compile(LNLargeCompile().eval(), torch.randn(2, 32, 512))


def test_layernorm_large_backend():
    class LNLargeBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(512)

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_backend(LNLargeBackend().eval(), torch.randn(2, 32, 512))


# --- No affine ---


def test_layernorm_no_affine_compile():
    class LNNoAffineCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(64, elementwise_affine=False)

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_compile(LNNoAffineCompile().eval(), torch.randn(4, 8, 64))


def test_layernorm_no_affine_backend():
    class LNNoAffineBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(64, elementwise_affine=False)

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_backend(LNNoAffineBackend().eval(), torch.randn(4, 8, 64))


# --- 2-D input (batch, features) ---


def test_layernorm_2d_input_compile():
    class LN2dInputCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(64)

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_compile(LN2dInputCompile().eval(), torch.randn(8, 64))


def test_layernorm_2d_input_backend():
    class LN2dInputBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(64)

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_backend(LN2dInputBackend().eval(), torch.randn(8, 64))


# --- 4-D input (batch, C, H, W) — normalize last 3 dims ---


def test_layernorm_4d_input_compile():
    class LN4dInputCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm([32, 4, 4])

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_compile(LN4dInputCompile().eval(), torch.randn(2, 32, 4, 4))


def test_layernorm_4d_input_backend():
    class LN4dInputBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm([32, 4, 4])

        def forward(self, x: torch.Tensor):
            return self.ln(x)

    check_backend(LN4dInputBackend().eval(), torch.randn(2, 32, 4, 4))
