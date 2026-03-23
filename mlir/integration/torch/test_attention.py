import copy

import pytest

import torch
import torch.nn as nn

import docc.torch


# --- helpers ---


def _check(model, *inputs, rtol=1e-4, atol=1e-5):
    model_ref = copy.deepcopy(model)
    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(*inputs)
        ref = model_ref(*inputs)
    assert torch.allclose(res, ref, rtol=rtol, atol=atol)


def _check_backend(model, *inputs, rtol=1e-4, atol=1e-5):
    docc.torch.set_backend_options(target="none", category="server")
    _check(model, *inputs, rtol=rtol, atol=atol)


# --- Self-attention: 8 heads, dim 64 ---


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_self_attention_compile():
    class SelfAttn8h64dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    _check(SelfAttn8h64dCompile().eval(), torch.randn(2, 16, 64))


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_self_attention_backend():
    class SelfAttn8h64dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    _check_backend(SelfAttn8h64dBackend().eval(), torch.randn(2, 16, 64))


# --- Self-attention: 4 heads, dim 128 ---


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_self_attention_4h128d_compile():
    class SelfAttn4h128dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=128, num_heads=4, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    _check(SelfAttn4h128dCompile().eval(), torch.randn(2, 32, 128))


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_self_attention_4h128d_backend():
    class SelfAttn4h128dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=128, num_heads=4, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    _check_backend(SelfAttn4h128dBackend().eval(), torch.randn(2, 32, 128))


# --- Self-attention: 1 head (degenerate) ---


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_self_attention_1head_compile():
    class SelfAttn1hCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=32, num_heads=1, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    _check(SelfAttn1hCompile().eval(), torch.randn(4, 8, 32))


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_self_attention_1head_backend():
    class SelfAttn1hBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=32, num_heads=1, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    _check_backend(SelfAttn1hBackend().eval(), torch.randn(4, 8, 32))


# --- Self-attention: 16 heads, dim 256 ---


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_self_attention_16h256d_compile():
    class SelfAttn16h256dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=256, num_heads=16, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    _check(SelfAttn16h256dCompile().eval(), torch.randn(1, 16, 256))


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_self_attention_16h256d_backend():
    class SelfAttn16h256dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=256, num_heads=16, batch_first=True
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    _check_backend(SelfAttn16h256dBackend().eval(), torch.randn(1, 16, 256))


# --- Self-attention: batch_first=False ---


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_self_attention_seq_first_compile():
    class SelfAttnSeqFirstCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=False
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    # (seq, batch, embed)
    _check(SelfAttnSeqFirstCompile().eval(), torch.randn(16, 2, 64))


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_self_attention_seq_first_backend():
    class SelfAttnSeqFirstBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=False
            )

        def forward(self, x: torch.Tensor):
            out, _ = self.attn(x, x, x)
            return out

    _check_backend(SelfAttnSeqFirstBackend().eval(), torch.randn(16, 2, 64))


# --- Cross-attention ---


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_cross_attention_compile():
    class CrossAttn8h64dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=True
            )

        def forward(self, q: torch.Tensor, kv: torch.Tensor):
            out, _ = self.attn(q, kv, kv)
            return out

    _check(
        CrossAttn8h64dCompile().eval(), torch.randn(2, 8, 64), torch.randn(2, 16, 64)
    )


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_cross_attention_backend():
    class CrossAttn8h64dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=64, num_heads=8, batch_first=True
            )

        def forward(self, q: torch.Tensor, kv: torch.Tensor):
            out, _ = self.attn(q, kv, kv)
            return out

    _check_backend(
        CrossAttn8h64dBackend().eval(), torch.randn(2, 8, 64), torch.randn(2, 16, 64)
    )


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_cross_attention_large_compile():
    class CrossAttn4h128dCompile(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=128, num_heads=4, batch_first=True
            )

        def forward(self, q: torch.Tensor, kv: torch.Tensor):
            out, _ = self.attn(q, kv, kv)
            return out

    _check(
        CrossAttn4h128dCompile().eval(), torch.randn(1, 4, 128), torch.randn(1, 64, 128)
    )


@pytest.mark.skip(reason="requires tensor.extract_slice")
def test_cross_attention_large_backend():
    class CrossAttn4h128dBackend(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=128, num_heads=4, batch_first=True
            )

        def forward(self, q: torch.Tensor, kv: torch.Tensor):
            out, _ = self.attn(q, kv, kv)
            return out

    _check_backend(
        CrossAttn4h128dBackend().eval(), torch.randn(1, 4, 128), torch.randn(1, 64, 128)
    )
