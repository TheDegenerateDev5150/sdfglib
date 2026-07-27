import pytest
import torch
import torch.nn as nn

from tests import check


# --- Test Models ---
class EqTensorNet(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.eq(x, y)


class EqScalarNet(nn.Module):
    def __init__(self, scalar_value: float | int):
        super().__init__()
        self.scalar_value = scalar_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.eq(x, self.scalar_value)


# --- Test Data Generators ---
def generate_tensor_pair(
    shape: tuple[int, ...], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a pair of tensors with a roughly 50% mix of equal elements."""
    torch.manual_seed(42)
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype)
        y = torch.randn(shape, dtype=dtype)
    else:
        x = torch.randint(-5, 5, shape, dtype=dtype)
        y = torch.randint(-5, 5, shape, dtype=dtype)

    # Force roughly half of the elements to be exactly equal
    mask = torch.rand(shape) > 0.5
    y[mask] = x[mask]
    return x, y


def generate_tensor_and_scalar(
    shape: tuple[int, ...], dtype: torch.dtype
) -> tuple[torch.Tensor, float | int]:
    """Generates a tensor and a scalar with a roughly 50% mix of equal elements."""
    torch.manual_seed(42)
    scalar_val = 2.0 if dtype.is_floating_point else 2

    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype)
    else:
        x = torch.randint(-5, 5, shape, dtype=dtype)

    # Force roughly half of the elements to equal the scalar
    mask = torch.rand(shape) > 0.5
    x[mask] = scalar_val
    return x, scalar_val


# --- Test Suite ---
SHAPES = [(4,), (3, 4), (2, 3, 4)]
DTYPES = [torch.int64, torch.float32, torch.float64]


class TestEq:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_tensor_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype, target: str
    ) -> None:
        torch._dynamo.reset()
        x, y = generate_tensor_pair(shape, dtype)
        check(EqTensorNet(), x, y, target=target)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_tensor_scalar(
        self, shape: tuple[int, ...], dtype: torch.dtype, target: str
    ) -> None:
        torch._dynamo.reset()
        x, scalar_val = generate_tensor_and_scalar(shape, dtype)
        check(EqScalarNet(scalar_val), x, target=target)
