import pytest

import torch
import torch.nn as nn

from docc.torch import compile_torch


def test_add():
    class AddNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.add(x, x)

    model = AddNet()
    model_ref = AddNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


def test_add2():
    class Add2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.add(torch.add(x, x), x)

    model = Add2Net()
    model_ref = Add2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


def test_sub():
    class SubNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sub(x, x)

    model = SubNet()
    model_ref = SubNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


def test_sub2():
    class Sub2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sub(torch.sub(x, x), x)

    model = Sub2Net()
    model_ref = Sub2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


def test_mul():
    class MulNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.mul(x, x)

    model = MulNet()
    model_ref = MulNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


def test_mul2():
    class Mul2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.mul(torch.mul(x, x), x)

    model = Mul2Net()
    model_ref = Mul2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


def test_div():
    class DivNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.div(x, x)

    model = DivNet()
    model_ref = DivNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


def test_div2():
    class Div2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.div(torch.div(x, x), x)

    model = Div2Net()
    model_ref = Div2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_abs():
    class AbsNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.abs(x)

    model = AbsNet()
    model_ref = AbsNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_abs2():
    class Abs2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.abs(torch.abs(x))

    model = Abs2Net()
    model_ref = Abs2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_ceil():
    class CeilNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.ceil(x)

    model = CeilNet()
    model_ref = CeilNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_ceil2():
    class Ceil2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.ceil(torch.ceil(x))

    model = Ceil2Net()
    model_ref = Ceil2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_erf():
    class ErfNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.erf(x)

    model = ErfNet()
    model_ref = ErfNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_erf2():
    class Erf2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.erf(torch.erf(x))

    model = Erf2Net()
    model_ref = Erf2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


def test_exp():
    class ExpNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.exp(x)

    model = ExpNet()
    model_ref = ExpNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


def test_exp2():
    class Exp2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.exp(torch.exp(x))

    model = Exp2Net()
    model_ref = Exp2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_floor():
    class FloorNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.floor(x)

    model = FloorNet()
    model_ref = FloorNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_floor2():
    class Floor2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.floor(torch.floor(x))

    model = Floor2Net()
    model_ref = Floor2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_log():
    class LogNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.log(x)

    model = LogNet()
    model_ref = LogNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_log2():
    class Log2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.log(torch.log(x))

    model = Log2Net()
    model_ref = Log2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_max():
    class MaxNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.max(x, x)

    model = MaxNet()
    model_ref = MaxNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_max2():
    class Max2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.max(torch.max(x, x), x)

    model = Max2Net()
    model_ref = Max2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_min():
    class MinNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.min(x, x)

    model = MinNet()
    model_ref = MinNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_min2():
    class Min2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.min(torch.min(x, x), x)

    model = Min2Net()
    model_ref = Min2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_pow():
    class PowNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.pow(x, 3)

    model = PowNet()
    model_ref = PowNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_pow2():
    class Pow2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.pow(torch.pow(x, 2), 2)

    model = Pow2Net()
    model_ref = Pow2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_round():
    class RoundNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.round(x)

    model = RoundNet()
    model_ref = RoundNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_round2():
    class Round2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.round(torch.round(x))

    model = Round2Net()
    model_ref = Round2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_sqrt():
    class SqrtNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sqrt(x)

    model = SqrtNet()
    model_ref = SqrtNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_sqrt2():
    class Sqrt2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.sqrt(torch.sqrt(x))

    model = Sqrt2Net()
    model_ref = Sqrt2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_tanh():
    class TanhNet(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.tanh(x)

    model = TanhNet()
    model_ref = TanhNet()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)


@pytest.mark.skip()
def test_tanh2():
    class Tanh2Net(nn.Module):
        def forward(self, x: torch.Tensor):
            return torch.tanh(torch.tanh(x))

    model = Tanh2Net()
    model_ref = Tanh2Net()
    example_input = torch.randn(3, 4)

    program = compile_torch(model, example_input)
    res = program(example_input)

    ref = model_ref(example_input)
    assert torch.allclose(res, ref, rtol=1e-5)
